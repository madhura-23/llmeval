"""
LLMEval CLI

Usage:
    llmeval run --config configs/example.yaml
    llmeval run --config configs/example.yaml --output results/
    llmeval validate --config configs/example.yaml
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()  # automatically loads .env from project root

app = typer.Typer(name="llmeval", help="LLM evaluation & benchmarking framework", add_completion=False)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Load config & dataset only, no API calls"),
):
    """Run an evaluation as defined in the YAML config."""
    from llmeval.config import EvalConfig
    from llmeval.runner import EvalRunner

    try:
        cfg = EvalConfig.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(code=1)

    if dry_run:
        from llmeval.dataset import DatasetLoader
        loader = DatasetLoader(cfg.dataset)
        samples = loader.load()
        console.print(f"[green]✓ Dry run OK[/green] — {len(samples)} samples, {len(cfg.models)} model(s)")
        return

    runner = EvalRunner(cfg)
    results = asyncio.run(runner.run())
    runner.save_results(results, output_dir=str(output) if output else None)


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config file", exists=True),
):
    """Validate a YAML config without running any evaluations."""
    from llmeval.config import EvalConfig

    try:
        cfg = EvalConfig.from_yaml(config)
        console.print(f"[green]✓ Config valid[/green] — {len(cfg.models)} model(s), metrics: {cfg.metrics}")
    except Exception as e:
        console.print(f"[red]✗ Config invalid:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


@app.command()
def compare(
    results_dir: Path = typer.Option(..., "--dir", "-d", help="Directory containing result JSON files"),
    baseline: str = typer.Option(None, "--baseline", "-b", help="Baseline model name"),
    output: Path = typer.Option(None, "--output", "-o", help="Save comparison JSON to this path"),
):
    """Compare results across models from a results directory."""
    import glob, json as _json
    from llmeval.runner import EvalResult, SampleResult
    from llmeval.comparison import compare_results
    from llmeval.reporter import Reporter

    files = sorted(glob.glob(str(results_dir / "*.json")))
    if not files:
        console.print(f"[red]No JSON files found in {results_dir}[/red]")
        raise typer.Exit(1)

    # Load the most recent result file
    latest = files[-1]
    with open(latest) as f:
        raw = _json.load(f)

    # Reconstruct EvalResult objects
    results = []
    for entry in raw:
        r = EvalResult(
            run_name=entry["run_name"],
            model_name=entry["model_name"],
            model_id=entry["model_id"],
            provider=entry["provider"],
            total_samples=entry["total_samples"],
            errors=entry["errors"],
            duration_s=entry["duration_s"],
            total_cost_usd=entry["total_cost_usd"],
            total_input_tokens=entry["total_input_tokens"],
            total_output_tokens=entry["total_output_tokens"],
            aggregate_scores=entry["aggregate_scores"],
        )
        for s in entry.get("samples", []):
            r.sample_results.append(SampleResult(
                sample_id=s["id"],
                model_name=entry["model_name"],
                input=s["input"],
                reference=s["reference"],
                prediction=s["prediction"],
                scores=s.get("scores", {}),
                latency_ms=s.get("latency_ms", 0),
                cost_usd=s.get("cost_usd", 0),
                error=s.get("error"),
            ))
        results.append(r)

    report = compare_results(results, baseline_model=baseline)
    reporter = Reporter(report)
    reporter.print_all()

    if output:
        reporter.save_json(output)
    else:
        default_out = results_dir / f"comparison_{int(__import__('time').time())}.json"
        reporter.save_json(default_out)
