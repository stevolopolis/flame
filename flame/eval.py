
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

        self.seq_lens = [2**i for i in range(8, 20) if 2 ** i <= self.job_config.eval.seq_len]
        self.val_steps = job_config.eval.steps
        self.device = "cuda"

        self.dataloader = self._build_val_split_dataloader()

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

                with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
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





if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()

    # Check if model safetensor exists
    # if not os.path.exists(f"{config.job.dump_folder}/model.safetensors"):
    logger.info(f"Model safetensor not found for {config.job.dump_folder}")
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
    metrics, losses = eval_manager.eval()

    # Create save directory
    save_dir = f"{config.job.dump_folder}/eval" 
    os.makedirs(save_dir, exist_ok=True)

    # Store the losses to a .npy file
    np.save(f"{save_dir}/losses.npy", losses.cpu().numpy())

    per_token_nll_plots = eval_manager.plot_per_token_nll(metrics)
    log_perplexity_plot = eval_manager.plot_log_perplexity(metrics)

    for seq_len, img in per_token_nll_plots.items():
        # save image to file
        img.save(f"{save_dir}/per_token_nll_{seq_len}.png")

    log_perplexity_plot.save(f"{save_dir}/log_perplexity.png")
