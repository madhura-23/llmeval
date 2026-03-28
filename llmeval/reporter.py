"""
Reporter — renders ComparisonReport as Rich terminal tables and JSON.

Usage:
    report = compare_results(results)
    reporter = Reporter(report)
    reporter.print_score_matrix()
    reporter.print_metric_winners()
    reporter.save_json("results/comparison.json")
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

from llmeval.comparison import ComparisonReport

console = Console()


class Reporter:
    def __init__(self, report: ComparisonReport):
        self.report = report

    def print_all(self) -> None:
        """Print the full comparison report to terminal."""
        self._print_header()
        self.print_score_matrix()
        self.print_metric_winners()
        self.print_delta_table()

    def _print_header(self) -> None:
        r = self.report
        console.print(f"\n[bold]📊 Comparison Report[/bold] — [cyan]{r.run_name}[/cyan]")
        console.print(
            f"   {r.sample_count} samples · {len(r.models)} models · "
            f"{len(r.metrics)} metrics · baseline: [yellow]{r.baseline_model}[/yellow]\n"
        )

    def print_score_matrix(self) -> None:
        """Print model × metric score matrix."""
        r = self.report
        table = Table(
            title="Score matrix",
            box=box.SIMPLE_HEAD,
            show_lines=True,
            title_style="bold",
        )
        table.add_column("Model", style="bold cyan", min_width=16)
        for metric in r.metrics:
            table.add_column(metric, justify="right", min_width=10)
        table.add_column("Overall", justify="right", style="bold green", min_width=10)

        # Find best score per metric for highlighting
        best_per_metric = {
            m: max(r.score_matrix[model][m] for model in r.models)
            for m in r.metrics
        }

        for model in r.models:
            row = [model]
            for metric in r.metrics:
                score = r.score_matrix[model][metric]
                is_best = abs(score - best_per_metric[metric]) < 1e-6
                cell = f"[bold green]{score:.4f}[/bold green]" if is_best else f"{score:.4f}"
                row.append(cell)
            row.append(f"{r.overall_scores[model]:.4f}")
            table.add_row(*row)

        console.print(table)

    def print_metric_winners(self) -> None:
        """Print winner per metric."""
        r = self.report
        table = Table(title="Winners per metric", box=box.SIMPLE_HEAD, title_style="bold")
        table.add_column("Metric", style="bold")
        table.add_column("Winner", style="green")
        table.add_column("Score", justify="right")
        table.add_column("Runner-up", style="yellow")
        table.add_column("Gap", justify="right")

        for mc in r.metric_comparisons:
            sorted_models = sorted(mc.scores, key=mc.scores.get, reverse=True)
            winner = sorted_models[0]
            runner_up = sorted_models[1] if len(sorted_models) > 1 else "—"
            gap = (
                mc.scores[winner] - mc.scores[runner_up]
                if runner_up != "—" else 0.0
            )
            table.add_row(
                mc.metric,
                winner,
                f"{mc.scores[winner]:.4f}",
                runner_up,
                f"{gap:+.4f}",
            )

        console.print(table)
        console.print(
            f"\n🏆 [bold green]Overall winner:[/bold green] "
            f"[cyan]{r.overall_winner}[/cyan] "
            f"(mean score: {r.overall_scores[r.overall_winner]:.4f})\n"
        )

    def print_delta_table(self) -> None:
        """Print delta vs baseline with significance indicators."""
        r = self.report
        non_baseline = [m for m in r.models if m != r.baseline_model]
        if not non_baseline:
            return

        table = Table(
            title=f"Delta vs baseline ({r.baseline_model})",
            box=box.SIMPLE_HEAD,
            title_style="bold",
        )
        table.add_column("Metric", style="bold")
        for model in non_baseline:
            table.add_column(model, justify="right", min_width=14)

        for mc in r.metric_comparisons:
            row = [mc.metric]
            for model in non_baseline:
                delta = mc.deltas.get(model, 0.0)
                sig = mc.significant.get(model, False)
                star = " ★" if sig else ""
                color = "green" if delta > 0 else "red" if delta < 0 else "white"
                row.append(f"[{color}]{delta:+.4f}{star}[/{color}]")
            table.add_row(*row)

        console.print(table)
        console.print("  [dim]★ = statistically significant (p < 0.05, bootstrap)[/dim]\n")

    def save_json(self, path: str | Path) -> Path:
        """Export comparison report to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        r = self.report
        payload = {
            "run_name": r.run_name,
            "baseline_model": r.baseline_model,
            "overall_winner": r.overall_winner,
            "overall_scores": r.overall_scores,
            "sample_count": r.sample_count,
            "models": r.models,
            "metrics": r.metrics,
            "score_matrix": r.score_matrix,
            "metric_comparisons": [
                {
                    "metric": mc.metric,
                    "scores": mc.scores,
                    "winner": mc.winner,
                    "deltas": mc.deltas,
                    "significant": mc.significant,
                }
                for mc in r.metric_comparisons
            ],
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)

        console.print(f"[green]Comparison saved →[/green] {out}")
        return out
