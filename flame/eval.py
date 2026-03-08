
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from fla.ops.utils import prepare_position_ids
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import os
from typing import *
import logging
from tqdm import tqdm
from dataclasses import dataclass

import custom_models
from custom_models.lact_model.layer_lact_swiglu import LaCTSWIGLULayer
from flame.data import build_dataloader, build_dataset
from flame.config_manager import JobConfig
from flame.utils.convert_dcp_to_hf import save_pretrained

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

torch.set_num_threads(16)

class EvalManager:
    """
    EvalManager is responsible for evaluating the model during training.
    """
    def __init__(
        self,
        job_config: JobConfig,
        model: AutoModelForCausalLM,
        states: dict[str, Any],
    ) -> None:
        self.job_config = job_config
        self.model = model
        self.states = states

        self.seq_lens = [2**i for i in range(6, 20) if 2 ** i <= self.job_config.eval.seq_len]
        self.val_steps = job_config.eval.steps
        self.device = "cuda"

        self.dataloader = self._build_val_split_dataloader()

    def activate_model_state_tracking(self):
        """
        For each LaCTSWIGLULayer in the model, set the track_states attribute to True.
        """
        for module in self.model.modules():
            if isinstance(module, LaCTSWIGLULayer):
                logger.info(f"Activating model state tracking for {module}")
                module.track_states = True

    def _build_val_split_dataloader(self):
        dp_degree, dp_rank = 1, 0
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.job_config.model.tokenizer_path,
            trust_remote_code=True,
            model_max_length=int(1e10),
        )
        logger.info(f"{tokenizer}")
        logger.info(
            f"Loading dataset {self.job_config.eval.dataset}"
            f":{self.job_config.eval.dataset_name}"
            f":{self.job_config.eval.dataset_split}"
            f":{self.job_config.eval.data_dir}"
            f":{self.job_config.eval.data_files}"
            if self.job_config.eval.dataset_name is not None
            else ""
        )
        dataset = build_dataset(
            dataset=self.job_config.eval.dataset,
            dataset_name=self.job_config.eval.dataset_name,
            dataset_split=self.job_config.eval.dataset_split,
            data_dir=self.job_config.eval.data_dir,
            data_files=self.job_config.eval.data_files,
            data_probs=self.job_config.eval.data_probs,
            streaming=self.job_config.eval.streaming,
            dp_degree=dp_degree,
            num_workers=self.job_config.eval.num_workers,
            seed=self.job_config.eval.seed,
        )

        logger.info("Building dataloader...")
        dataloader = build_dataloader(
            dataset=dataset,
            tokenizer=tokenizer,
            rank=dp_rank,
            world_size=dp_degree,
            batch_size=self.job_config.eval.batch_size,
            seq_len=self.job_config.eval.seq_len,
            context_len=self.job_config.eval.context_len,
            varlen=self.job_config.eval.varlen,
            num_workers=self.job_config.eval.num_workers,
            pin_memory=self.job_config.eval.pin_memory,
            persistent_workers=self.job_config.eval.persistent_workers,
            snapshot_every_n_steps=self.job_config.checkpoint.interval,
        )

        return iter(dataloader)

    def _prepare_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        cu_seqlens = batch["cu_seqlens"].to(self.device) if "cu_seqlens" in batch else None
        if cu_seqlens is not None:
            position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
        else:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=self.device)
                .repeat(input_ids.shape[0], 1)
                .to(torch.int32)
            )

        return input_ids, labels, cu_seqlens, position_ids

    def eval(self) -> None:
        """
        Evaluate the model during training.

        Included evals:
            1. Validation split of long-data-collections-without-books

        Evaluation metrics:
            1. Log perplexity
            2. Per-token NLL
        """
        logger.info("Evaluating the model...")

        losses = []
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        self.model.eval()
        with torch.inference_mode():
            for step in tqdm(range(self.val_steps)):
                batch = next(self.dataloader)
                input_ids, labels, cu_seqlens, position_ids = self._prepare_inputs(batch)

                with torch.autocast(device_type=self.device, enabled=True, dtype=torch.bfloat16):
                    output = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                    )

                # calculate per-token NLL
                labels = torch.cat(
                    (
                        labels[..., 1:],
                        torch.full_like(labels[:, :1], criterion.ignore_index),
                    ),
                    1,
                )
                labels = labels.detach()
                logits = output.logits.detach()
                per_token_nll = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
                per_token_nll = per_token_nll.view(labels.shape[0], labels.shape[1])
                losses.append(per_token_nll.detach().cpu().to(torch.float32))

        losses = torch.cat(losses, dim=0)

        metrics = {"long-data-collections-val-split": self._calculate_metrics(losses)}

        return metrics, losses

    def _calculate_metrics(self, losses: torch.Tensor) -> dict:
        """
        Calculate the metrics.
        """
        # Per-token NLL
        per_token_nll_mean = torch.mean(losses, dim=0)
        per_token_nll_std = torch.std(losses, dim=0)
        # Log perplexity
        log_perplexity = {seq_len: torch.mean(losses[:, :seq_len]) for seq_len in self.seq_lens}

        return {
            "Per-token NLL Mean": per_token_nll_mean,
            "Per-token NLL Std": per_token_nll_std,
            "Log Perplexity": log_perplexity,
        }

    def plot_log_perplexity(
        self,
        metrics: dict,
    ) -> None:
        """
        Plot the log perplexity.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(
            metrics["long-data-collections-val-split"]["Log Perplexity"].keys(),
            metrics["long-data-collections-val-split"]["Log Perplexity"].values(),
            label="Log Perplexity",
        )
        plt.legend()
        plt.xlabel("Sequence Length")
        plt.ylabel("Log Perplexity")
        plt.title("Log Perplexity at different sequence lengths")
        plt.grid(True)
        plt.xscale("log", base=2)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        img = Image.open(img)
        plt.close()

        return img

    def plot_per_token_nll(
        self,
        metrics: dict,
    ) -> None:
        """
        Plot the per-token NLL with std.
        """
        per_token_nll_mean_full = metrics["long-data-collections-val-split"]["Per-token NLL Mean"]
        per_token_nll_std_full = metrics["long-data-collections-val-split"]["Per-token NLL Std"]

        imgs = {}
        for seq_len in self.seq_lens:
            per_token_nll_mean = per_token_nll_mean_full[:min(seq_len, len(per_token_nll_mean_full) - 1)]
            per_token_nll_std = per_token_nll_std_full[:min(seq_len, len(per_token_nll_std_full) - 1)]

            plt.figure(figsize=(10, 5))
            plt.plot(per_token_nll_mean.cpu().numpy(), label="Mean")
            plt.fill_between(range(len(per_token_nll_mean)), per_token_nll_mean.cpu().numpy() - per_token_nll_std.cpu().numpy(), per_token_nll_mean.cpu().numpy() + per_token_nll_std.cpu().numpy(), alpha=0.2)
            plt.hlines(y=np.min(per_token_nll_mean.cpu().numpy()), xmin=0, xmax=len(per_token_nll_mean), colors="red", linestyles="dashed", label="Min")
            plt.legend()

            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            img = Image.open(img)

            imgs[seq_len] = img

        return imgs

    def plot_model_states(self):
        imgs = {}
        for name, module in self.model.named_modules():
            imgs[name] = {"Matrix 0": {}, "Matrix 1": {}, "Matrix 2": {}}
            if isinstance(module, LaCTSWIGLULayer):
                if module.track_states:
                    # Reg lr
                    # imgs[name]["Matrix 0"].update(self.plot_model_regs(module.fw_reg_lr1.cpu().numpy()))
                    # imgs[name]["Matrix 1"].update(self.plot_model_regs(module.fw_reg_lr2.cpu().numpy()))
                    # imgs[name]["Matrix 2"].update(self.plot_model_regs(module.fw_reg_lr3.cpu().numpy()))

                    # W_t norm
                    imgs[name]["Matrix 0"].update(self.plot_model_norms(module.fw_w0_norm.cpu().numpy()))
                    imgs[name]["Matrix 1"].update(self.plot_model_norms(module.fw_w1_norm.cpu().numpy()))
                    imgs[name]["Matrix 2"].update(self.plot_model_norms(module.fw_w2_norm.cpu().numpy()))

                    # W_t - W_0 dist
                    imgs[name]["Matrix 0"].update(self.plot_model_dists(module.fw_w0_dist.cpu().numpy()))
                    imgs[name]["Matrix 1"].update(self.plot_model_dists(module.fw_w1_dist.cpu().numpy()))
                    imgs[name]["Matrix 2"].update(self.plot_model_dists(module.fw_w2_dist.cpu().numpy()))
                    break
                else:
                    logger.info(f"Model state tracking is not active for {name}")

        return imgs

    def plot_model_regs(self, reg_lr: torch.Tensor):
        f"""
        Plot the trend and distribution of the regularization scalars.

        Args:
            reg_lr: [b, nh, T]

        Returns:
            imgs: dict of images. Two plots per head. One for the trend and one for the distribution.
        """
        B, H, L = reg_lr.shape
        imgs = {"Reg LR Trend": {}, "Reg LR Distribution": {}}
        for h in range(H):
            reg_lr_h = reg_lr[0, h, :]

            plt.figure(figsize=(10, 5))
            plt.plot(reg_lr_h, label="Reg Lr")
            plt.legend()
            plt.xlabel("Sequence Length")
            plt.ylabel("Reg Lr")
            plt.title(f"Reg Lr for head {h}")
            plt.grid(True)
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            img = Image.open(img)
            imgs["Reg LR Trend"][f"Head {h}"] = img

            plt.close()

            plt.figure(figsize=(10, 5))
            plt.hist(np.mean(reg_lr[:, h, :], axis=0), bins=10, label="Reg Lr")
            plt.legend()
            plt.xlabel("Reg Lr")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of Reg Lr for head {h}")
            plt.grid(True)
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            img = Image.open(img)
            imgs["Reg LR Distribution"][f"Head {h}"] = img

            plt.close()

        return imgs

    def plot_model_norms(self, norm: torch.Tensor):
        """
        Plot the L2 norm of the model weights.

        Args:
            norm: [b, nh, T]

        Returns:
            imgs: dict of images. One plot per head.
        """
        B, H, T = norm.shape
        imgs = {"L2(W_t)": {}}
        for h in range(H):
            norm_h = norm[0, h, :]

            plt.figure(figsize=(10, 5))
            plt.plot(norm_h, label="Norm")
            plt.legend()
            plt.xlabel("Sequence Length")
            plt.ylabel("Norm")
            plt.title(f"Norm for head {h}")
            plt.grid(True)
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format="png")
            img.seek(0)
            img = Image.open(img)
            imgs["L2(W_t)"][f"Head {h}"] = img

            plt.close()

        return imgs

    def plot_model_dists(self, dist: torch.Tensor):
        """
        Plot the trend of model distances.

        Args:
            dist: [b, nh, T]

        Returns:
            imgs: dict of images. One plot per head.
        """
        B, H, T = dist.shape
        imgs = {"L2(W_t - W_0)": {}}
        for h in range(H):
            dist_h = dist[0, h, :]
            plt.figure(figsize=(10, 5)) 
            plt.plot(dist_h, label="Dist")
            plt.legend()
            plt.xlabel("Sequence Length")
            plt.ylabel("Dist")
            plt.title(f"Dist for head {h}")
            plt.grid(True)
            plt.tight_layout()
            img = io.BytesIO()  
            plt.savefig(img, format="png")
            img.seek(0)
            img = Image.open(img)
            imgs["L2(W_t - W_0)"][f"Head {h}"] = img

            plt.close()

        return imgs


def debugging_configs(config: JobConfig):
    config.job.config_file = "flame/models/fla.toml"
    config.job.dump_folder = "/scratch/sl12886/ttt/flame/exp/lact/lact-wregLinear-fixed-raw-nonorm-c32w32-bs512-seqlen256-context256-l4d256/run0-lr1e-2-seed42"
    config.model.config = "/scratch/sl12886/ttt/flame/configs/XS_lact_swiglu_nh4_fwlow_rank_nonorm_wregLinear0.1.json"
    config.model.tokenizer_path = "fla-hub/transformer-1.3B-100B"
    config.eval.batch_size = 8
    config.eval.seq_len = 8192
    config.eval.context_len = 8192
    config.eval.steps = 1
    config.eval.dataset = "arrow"
    config.eval.dataset_split = "train"
    config.eval.data_dir = "/scratch/sl12886/data/long-data-collections-val"
    config.eval.pin_memory = False
    config.eval.persistent_workers = False
    config.eval.prefetch_factor = 2
    config.eval.seed = 42
    config.eval.varlen = False

if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()

    if config.eval.debug:
        debugging_configs(config)
        logger.info(f"Config: {config}")

    # Find the latest checkpoint
    ckpt_path = f"{config.job.dump_folder}/checkpoint"
    ckpt = max([name.split("-")[-1] for name in os.listdir(ckpt_path)])
    # Convert the latest checkpoint to safetensor
    logger.info(f"Converting the latest checkpoint {ckpt} to safetensor...")
    save_pretrained(config.job.dump_folder, ckpt, config.model.config, config.model.tokenizer_path)
    logger.info(f"Model safetensor saved to {config.job.dump_folder}/model.safetensors")

    # Check if eval losses.npy exists. If so, skip evaluation.
    #if os.path.exists(f"{config.job.dump_folder}/eval/losses.npy"):
    #    logger.info(f"Eval losses {config.job.dump_folder}/eval/losses.npy already exists. Skipping evaluation.")
    #    exit()

    logger.info(f"Loading model from {config.job.dump_folder}")
    test_model = AutoModelForCausalLM.from_pretrained(config.job.dump_folder, device_map="auto")

    eval_manager = EvalManager(config, test_model, None)
    if config.eval.debug:
        eval_manager.activate_model_state_tracking()
    metrics, losses = eval_manager.eval()

    # Create save directory
    save_dir = f"{config.job.dump_folder}/eval" 
    os.makedirs(save_dir, exist_ok=True)

    if config.eval.debug:
        os.makedirs(f"{save_dir}/model_states", exist_ok=True)
        logger.info(f"Plotting model states...")
        model_states_plots = eval_manager.plot_model_states()
        logger.info(f"Saving model states plots...")
        for module_name, module_plots in model_states_plots.items():
            for matrix_name, matrix_plots in module_plots.items():
                for plot_name, plots in matrix_plots.items():
                    for head, head_img in plots.items():
                        head_img.save(f"{save_dir}/model_states/{plot_name}_{module_name}_{matrix_name}_{head}.png")
    else:
        # Store the losses to a .npy file
        np.save(f"{save_dir}/losses.npy", losses.cpu().numpy())

        logger.info(f"Plotting per-token NLL...")
        per_token_nll_plots = eval_manager.plot_per_token_nll(metrics)
        logger.info(f"Plotting log perplexity...")
        log_perplexity_plot = eval_manager.plot_log_perplexity(metrics)

        logger.info(f"Saving per-token NLL plots...")
        for seq_len, img in per_token_nll_plots.items():
            # save image to file
            img.save(f"{save_dir}/per_token_nll_{seq_len}.png")

        logger.info(f"Saving log perplexity plot...")
        log_perplexity_plot.save(f"{save_dir}/log_perplexity.png")

        
        # Store the checkpoint used to a txt file for validation
        with open(f"{save_dir}/checkpoint.txt", "w") as f:
            f.write(ckpt)