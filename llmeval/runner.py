"""
EvalRunner — orchestrates evaluation across models and samples.

For each (model, sample) pair:
  1. Format the prompt from the template
  2. Call the model connector (async, bounded concurrency via semaphore)
  3. Pass the response to each enabled metric
  4. Accumulate results into an EvalResult

Usage:
    runner = EvalRunner(config)
    results = await runner.run()
    runner.save_results(results)
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from llmeval.config import EvalConfig
from llmeval.connectors import ModelResponse, build_connector
from llmeval.dataset import DatasetLoader, EvalSample

console = Console()


@dataclass
class SampleResult:
    """Evaluation result for a single (model, sample) pair."""

    sample_id: str
    model_name: str
    input: str
    reference: str
    prediction: str
    scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class EvalResult:
    """Aggregated results for a full evaluation run."""

    run_name: str
    model_name: str
    model_id: str
    provider: str
    total_samples: int
    sample_results: list[SampleResult] = field(default_factory=list)
    aggregate_scores: dict[str, float] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: int = 0
    duration_s: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        n = len(self.sample_results) - self.errors
        return self.total_latency_ms / n if n > 0 else 0.0


class EvalRunner:
    """
    Async evaluation runner.

    Loads the dataset, builds connectors, and evaluates all
    (model, sample) pairs with bounded concurrency.
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def run(self) -> list[EvalResult]:
        """Run the full evaluation and return results per model."""
        # Lazy import metrics here to avoid circular deps
        from llmeval.metrics import compute_metrics  # noqa: F401 — implemented in week 2

        cfg = self.config
        loader = DatasetLoader(cfg.dataset)
        samples = loader.load()

        console.print(
            f"\n[bold]LLMEval[/bold] · run: [cyan]{cfg.run_name}[/cyan] · "
            f"{len(samples)} samples · {len(cfg.models)} model(s)\n"
        )

        all_results: list[EvalResult] = []

        for model_cfg in cfg.models:
            console.print(f"[bold yellow]→ Evaluating[/bold yellow] {model_cfg.name} ({model_cfg.model_id})")
            connector = build_connector(model_cfg)
            result = await self._eval_model(model_cfg.name, connector, samples)
            all_results.append(result)

        self._print_leaderboard(all_results)
        # Auto-run comparison report when multiple models are evaluated
        if len(all_results) > 1 and any(r.aggregate_scores for r in all_results):
            from llmeval.comparison import compare_results
            from llmeval.reporter import Reporter
            report = compare_results(all_results)
            Reporter(report).print_all()
        return all_results

    async def _eval_model(self, model_name: str, connector, samples: list[EvalSample]) -> EvalResult:
        from llmeval.metrics import compute_metrics  # noqa: F401

        model_cfg = next(m for m in self.config.models if m.name == model_name)
        result = EvalResult(
            run_name=self.config.run_name,
            model_name=model_name,
            model_id=model_cfg.model_id,
            provider=model_cfg.provider.value,
            total_samples=len(samples),
        )

        t0 = time.perf_counter()

        with Progress(
            TextColumn("  [progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"  {model_name}", total=len(samples))

            async def process_sample(sample: EvalSample) -> SampleResult:
                async with self._semaphore:
                    return await self._run_sample(model_name, connector, sample)

            tasks = [process_sample(s) for s in samples]
            for coro in asyncio.as_completed(tasks):
                sr = await coro
                result.sample_results.append(sr)
                result.total_cost_usd += sr.cost_usd
                result.total_latency_ms += sr.latency_ms
                result.total_input_tokens += sr.input_tokens
                result.total_output_tokens += sr.output_tokens
                if sr.error:
                    result.errors += 1
                progress.advance(task)

        result.duration_s = time.perf_counter() - t0
        result.aggregate_scores = self._aggregate_scores(result.sample_results)

        console.print(
            f"  [green]✓[/green] Done in {result.duration_s:.1f}s · "
            f"cost: ${result.total_cost_usd:.4f} · errors: {result.errors}\n"
        )
        return result

    async def _run_sample(self, model_name: str, connector, sample: EvalSample) -> SampleResult:
        prompt = sample.format_prompt(self.config.prompt_template)
        sr = SampleResult(
            sample_id=sample.id,
            model_name=model_name,
            input=sample.input,
            reference=sample.reference,
            prediction="",
        )
        try:
            response: ModelResponse = await connector.generate(prompt)
            sr.prediction = response.text
            sr.latency_ms = response.latency_ms
            sr.cost_usd = response.cost_usd
            sr.input_tokens = response.input_tokens
            sr.output_tokens = response.output_tokens
            from llmeval.metrics import compute_metrics_async
            sr.scores = await compute_metrics_async(
                question=sample.input,
                reference=sample.reference,
                prediction=response.text,
                metrics_config=self.config.metrics,
                context=sample.context,
            )


        except Exception as exc:
            sr.error = str(exc)
            console.print(f"  [red]Error on sample {sample.id}:[/red] {exc}")
        return sr

    def _aggregate_scores(self, sample_results: list[SampleResult]) -> dict[str, float]:
        """Average all metric scores across non-errored samples."""
        valid = [sr for sr in sample_results if not sr.error and sr.scores]
        if not valid:
            return {}
        all_keys = set().union(*[sr.scores.keys() for sr in valid])
        return {
            key: sum(sr.scores.get(key, 0.0) for sr in valid) / len(valid)
            for key in all_keys
        }

    def save_results(self, results: list[EvalResult], output_dir: str | None = None) -> Path:
        """Write results to a timestamped JSON file."""
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"{self.config.run_name}_{int(time.time())}.json"

        payload = []
        for r in results:
            payload.append({
                "run_name": r.run_name,
                "model_name": r.model_name,
                "model_id": r.model_id,
                "provider": r.provider,
                "total_samples": r.total_samples,
                "errors": r.errors,
                "duration_s": r.duration_s,
                "total_cost_usd": r.total_cost_usd,
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
                "avg_latency_ms": r.avg_latency_ms,
                "aggregate_scores": r.aggregate_scores,
                "samples": [
                    {
                        "id": sr.sample_id,
                        "input": sr.input,
                        "reference": sr.reference,
                        "prediction": sr.prediction,
                        "scores": sr.scores,
                        "latency_ms": sr.latency_ms,
                        "cost_usd": sr.cost_usd,
                        "error": sr.error,
                    }
                    for sr in r.sample_results
                ],
            })

        with open(filename, "w") as f:
            json.dump(payload, f, indent=2)

        console.print(f"\n[green]Results saved →[/green] {filename}")
        return filename

    def _print_leaderboard(self, results: list[EvalResult]) -> None:
        """Print a Rich leaderboard table to the terminal."""
        table = Table(title=f"📊 Leaderboard — {self.config.run_name}", show_lines=True)
        table.add_column("Model", style="bold cyan")
        table.add_column("Samples", justify="right")
        table.add_column("Errors", justify="right", style="red")
        table.add_column("Avg latency", justify="right")
        table.add_column("Total cost", justify="right", style="yellow")
        table.add_column("Score", justify="right", style="green")

        for r in sorted(results, key=lambda x: sum(x.aggregate_scores.values()), reverse=True):
            agg = sum(r.aggregate_scores.values()) / max(len(r.aggregate_scores), 1)
            table.add_row(
                r.model_name,
                str(r.total_samples),
                str(r.errors),
                f"{r.avg_latency_ms:.0f}ms",
                f"${r.total_cost_usd:.4f}",
                f"{agg:.3f}" if r.aggregate_scores else "—",
            )

        console.print(table)
