import json
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import typer
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from rich.console import Console
from rich.table import Table

from dedup_tool.cli import get_strategy
from dedup_tool.config.settings import DedupConfig


app = typer.Typer()
console = Console()


@app.command()
def evaluate_pairs(
    eval_file: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Path to input file (overrides config)"
    ),
    config_file: Optional[Path] = typer.Option(Path("config.yaml"), "--config", "-c"),
):
    """Оценка качества дедупликации с использованием NumPy и Sklearn."""

    if not eval_file.exists():
        console.print(f"File not found: {eval_file}", style="red")
        raise typer.Exit(1)

    config = (
        DedupConfig.from_yaml(config_file) if config_file.exists() else DedupConfig()
    )
    logging.info(
        f"Loaded configuration\nstrategy: {config.algorithm}\nnum_perm: {config.num_perm}\nngram_size: {config.ngram_size}\nbands: {config.bands}"
    )
    console.print("[green]Using Custom implementation...[/green]")
    strategy = get_strategy(config)

    y_true_list = []
    y_pred_list = []
    num_lines = sum(1 for _ in open(eval_file, "r"))

    console.print(f"[bold blue]Starting evaluation on {eval_file}...[/bold blue]")

    with open(eval_file, "r") as f:
        for line in tqdm(f, total=num_lines, desc="Processing pairs", unit="pair"):
            data = json.loads(line)
            code1 = data["code1"]
            code2 = data["code2"]

            y_true_list.append(int(data["label"]))

            result = strategy.deduplicate([code1, code2])
            clusters = result.get("clusters", {})
            prediction = 0

            for members in clusters.values():
                if 0 in members and 1 in members:
                    prediction = 1
                    break

            y_pred_list.append(prediction)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    table = Table(title="Deduplication Metrics (Powered by Sklearn)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("True Positives (TP)", str(tp))
    table.add_row("False Positives (FP)", str(fp))
    table.add_row("False Negatives (FN)", str(fn))
    table.add_row("True Negatives (TN)", str(tn))
    table.add_section()
    table.add_row("Precision", f"{precision:.4f}")
    table.add_row("Recall", f"{recall:.4f}")
    table.add_row("F1-Score", f"{f1:.4f}")

    console.print(table)


if __name__ == "__main__":
    app()
