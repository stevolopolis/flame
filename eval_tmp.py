from dataclasses import dataclass
from typing import *
import os
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from pathlib import Path
import argparse

def pretty_print_model_statuses_rich(statuses: Dict[str, List[Any]]) -> None:
    """Print model statuses as a Rich table with colored booleans."""
    columns = [
        "model",
        "lr",
        "seed",
        "ckpt",
        "has_safetensors",
        "has_eval",
        "has_losses",
    ]
    if not all(col in statuses for col in columns):
        missing = [col for col in columns if col not in statuses]
        raise ValueError(f"Missing columns in statuses: {missing}")

    n_rows = len(statuses["model"])
    for col in columns:
        if len(statuses[col]) != n_rows:
            raise ValueError(f"Column '{col}' length does not match 'model' length")

    table = Table(title="Model Statuses", show_lines=False)
    table.add_column("Model", overflow="fold")
    table.add_column("LR", justify="right")
    table.add_column("Seed", justify="right")
    table.add_column("Ckpt", justify="right")
    table.add_column("Safetensors", justify="center")
    table.add_column("Eval Dir", justify="center")
    table.add_column("Losses", justify="center")

    for i in range(n_rows):
        lr_val = statuses["lr"][i]
        lr_str = f"{lr_val:.1e}" if isinstance(lr_val, (float, np.floating)) else str(lr_val)

        def rich_bool(v: Any) -> str:
            return "[green]True[/green]" if bool(v) else "[red]False[/red]"

        def rich_ckpt(v: Any) -> str:
            return f"[green]{v}[/green]" if 40960 else f"[red]{v}[/red]"

        table.add_row(
            str(statuses["model"][i]),
            lr_str,
            str(statuses["seed"][i]),
            str(statuses["ckpt"][i]),
            rich_bool(statuses["has_safetensors"][i]),
            rich_bool(statuses["has_eval"][i]),
            rich_bool(statuses["has_losses"][i]),
        )

    console = Console()
    console.print(table)


@dataclass
class EvalResults:
    model: str
    lr: float
    seed: int
    losses: Union[np.ndarray, str]

    def __post_init__(self) -> None:
        if isinstance(self.losses, str):
            self.losses = np.load(self.losses)

        self.metrics = self._calculate_metrics(self.losses)
        # Remove the raw losses array to save memory
        del self.losses

    def _calculate_metrics(self, losses: np.ndarray) -> dict:
        """
        Calculate the metrics.
        """
        # Per-token NLL
        per_token_nll_mean = np.mean(losses, axis=0)
        per_token_nll_std = np.std(losses, axis=0)
        # Log perplexity
        log_perplexity = {2**i: np.mean(losses[:, :2**i]) for i in range(8, 20) if 2 ** i <= losses.shape[1]}

        return {
            "Per-token NLL Mean": per_token_nll_mean,
            "Per-token NLL Std": per_token_nll_std,
            "Log Perplexity": log_perplexity,
        }


class EvalVisualizer:
    def __init__(
        self,
        results: List[EvalResults],
    ) -> None:
        self.results = results

    def filter_min_perplexity_per_model(self, ctx_len: int):
        """For each model, keep only the result with the minimum perplexity at ctx_len."""
        self.mins = {}
        for result in self.results:
            if result.metrics["Log Perplexity"] == {}:
                continue
            if ctx_len in result.metrics["Log Perplexity"]:
                perp = result.metrics["Log Perplexity"][ctx_len] 
            else:
                # get perp of longest sequence length
                perp = result.metrics["Log Perplexity"][max(result.metrics["Log Perplexity"].keys())]
            if result.model not in self.mins:
                self.mins[result.model] = (perp, result)
            
            if perp < self.mins[result.model][0]:
                self.mins[result.model] = (perp, result)
        
        self.results = [self.mins[model][1] for model in self.mins]

    def log_perplexity(self, train_len: int) -> plt.Figure:
        """
        Plot the log perplexity.
        """
        plt.figure(figsize=(10, 7.5))
        for result in self.results:
            plt.plot(
                result.metrics["Log Perplexity"].keys(),
                result.metrics["Log Perplexity"].values(),
                label=f"Model: {result.model}, LR: {result.lr}, Seed: {result.seed}",
            )
        plt.axvline(x=train_len, color="red", linestyle="--", label="Train Length")
        plt.legend()
        plt.xlabel("Sequence Length")
        plt.ylabel("Log Perplexity")
        plt.title("Log Perplexity at different sequence lengths")
        plt.grid(True)
        plt.xscale("log", base=2)
        plt.tight_layout()

        return plt.gcf()

    def per_token_nll(self, train_len: int, smoothed: bool = False) -> plt.Figure:
        """
        Plot the per-token NLL.
        """
        window = 64

        plt.figure(figsize=(10, 7.5))
        for result in self.results:
            curve = np.convolve(
                result.metrics["Per-token NLL Mean"][:-1],
                np.ones(window)/window, mode='valid'
            ) if smoothed else result.metrics["Per-token NLL Mean"][:-1]
            plt.plot(
                range(len(curve)),
                curve,
                label=f"Model: {result.model}, LR: {result.lr}, Seed: {result.seed}",
            )
        plt.axvline(x=train_len, color="red", linestyle="--", label="Train Length")
        plt.legend()
        plt.xlabel("Sequence Length")
        plt.ylabel("Per-token NLL")
        if smoothed:
            plt.title(f"Per-token NLL at different sequence lengths (Smoothed: w={window})")
        else:
            plt.title("Per-token NLL at different sequence lengths")
        plt.grid(True)
        plt.xscale("log", base=2)
        plt.tight_layout()

        return plt.gcf()


def get_model_status(path: str):
    """
    Returns the following status of a model:
        1. Latest checkpoint
        2. Whether it has model.safetensors
        3. Whether it has eval directory
        4. Whether it has losses.npy
    """
    ckpt_path = f"{path}/checkpoint"
    ckpt = max([name.split("-")[-1] for name in os.listdir(ckpt_path)])
    return ckpt, os.path.exists(f"{path}/model.safetensors"), os.path.exists(f"{path}/eval"), os.path.exists(f"{path}/eval/losses.npy")


def run_visualizer(results: List[EvalResults], save_dir: str, train_len: int = 2**8, filter_len: int = None) -> None:
    visualizer = EvalVisualizer(results)

    suffix = ""
    if filter_len is not None:
        visualizer.filter_min_perplexity_per_model(ctx_len=filter_len)
        suffix = f"At{filter_len}"

    visualizer.log_perplexity(train_len=train_len).savefig(f"{save_dir}/log_perplexity{suffix}.pdf")
    visualizer.per_token_nll(train_len=train_len).savefig(f"{save_dir}/per_token_nll{suffix}.pdf")
    visualizer.per_token_nll(train_len=train_len, smoothed=True).savefig(f"{save_dir}/per_token_nll_smoothed{suffix}.pdf")


if __name__ == "__main__":
    base_path = Path(f"/scratch/sl12886/ttt/flame/exp")

    statuses = {
        "model": [],
        "lr": [],
        "seed": [],
        "ckpt": [],
        "has_safetensors": [],
        "has_eval": [],
        "has_losses": [],
    }
    model_paths = []
    for p in base_path.rglob("run*lr*seed*"):
        if (p.is_dir() and "checkpoint" in os.listdir(p)):
            model_paths.append(p)

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    results = []
    results_per_model = {}
    
    for path in tqdm(model_paths):
        ckpt, has_safetensors, has_eval, has_losses = get_model_status(path)
        modelname = path.parent.name.replace("-bs512", "").replace("-context1024", "").replace("-context256", "")
        lr = path.name.split("lr")[-1].split("-seed")[0]
        seed = path.name.split("seed")[-1]
        statuses["model"].append(modelname)
        statuses["lr"].append(float(lr))
        statuses["seed"].append(int(seed))
        statuses["ckpt"].append(ckpt)
        statuses["has_safetensors"].append(has_safetensors)
        statuses["has_eval"].append(has_eval)
        statuses["has_losses"].append(has_losses)
        if not has_losses:
            continue

        eval_results = EvalResults(
            model=modelname, 
            lr=float(lr), 
            seed=int(seed), 
            losses=f"{path}/eval/losses.npy"
        )

        # Only keep a subset of models for the summary plots
        if (
            "1024" not in path.parent.name
        ):
            results.append(eval_results)
        if modelname not in results_per_model:
            results_per_model[modelname] = []
        results_per_model[modelname].append(eval_results)

    # Plot per model
    for modelname, r in results_per_model.items():
        save_dir_per_model = f"{save_dir}/{modelname}"
        os.makedirs(save_dir_per_model, exist_ok=True)
        run_visualizer(r, save_dir_per_model, train_len=2**8)
    # Summary plots (filter by train_len)
    run_visualizer(results, save_dir, train_len=2**8, filter_len=2**8)
    # Summary plots (filter by val_len)
    run_visualizer(results, save_dir, train_len=2**8, filter_len=2**14)

    # Pretty print the statuses of all models
    pretty_print_model_statuses_rich(statuses)