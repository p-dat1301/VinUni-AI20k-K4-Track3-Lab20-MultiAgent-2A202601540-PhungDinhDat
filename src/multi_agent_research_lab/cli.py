"""Command-line entrypoint for the lab."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    export_trace_json,
    log_langsmith_run,
    tracing_provider,
)
from multi_agent_research_lab.services.llm_client import build_llm_client
from multi_agent_research_lab.services.search_client import build_search_client

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()

DEFAULT_CONFIG = Path("configs/lab_default.yaml")


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _baseline_runner(query: str) -> ResearchState:
    """Single-agent baseline: one LLM call, no orchestration."""
    settings = get_settings()
    llm = build_llm_client(settings, settings.offline_corpus_dir)
    search = build_search_client(settings, settings.offline_corpus_dir)
    state = ResearchState(request=ResearchQuery(query=query))
    sources = search.search(query, state.request.max_sources)
    state.sources = sources
    context = "\n".join(
        f"- [{s.metadata.get('source_id') or f's{i}'}] {s.title}: {s.snippet[:300]}"
        for i, s in enumerate(sources, start=1)
    )
    response = llm.complete(
        "You are a research assistant. Answer the query concisely with evidence, "
        "citing sources as [source_id].",
        f"QUERY: {query}\nCONTEXT:\n{context}",
    )
    state.final_answer = response.content
    state.llm_provider = llm.provider
    state.record_usage(
        response.input_tokens or 0,
        response.output_tokens or 0,
        response.cost_usd or 0.0,
    )
    return state


def _multi_agent_runner(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    return MultiAgentWorkflow().run(state)


def _load_benchmark_queries(config_path: Path) -> list[str]:
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    queries = config.get("benchmark", {}).get("queries", [])
    return [str(q) for q in queries if q]


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline with a real LLM call."""
    _init()
    state = _baseline_runner(query)
    console.print(
        Panel.fit(
            state.final_answer or "(empty response)",
            title=f"Single-Agent Baseline ({state.llm_provider})",
            style="cyan",
        )
    )
    console.print(
        f"Tokens: {state.total_input_tokens + state.total_output_tokens} | "
        f"Cost: ${state.total_cost_usd:.6f}"
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    trace_output: Annotated[Path, typer.Option("--trace", help="Write trace JSON to path")] = Path(
        "reports/trace_multi_agent.json"
    ),
) -> None:
    """Run the multi-agent workflow end-to-end."""
    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    console.print(
        Panel.fit(result.final_answer or "(no answer)", title="Final Answer", style="green")
    )
    console.print(
        f"Route: {' -> '.join(result.route_history)} | Provider: {result.llm_provider} | "
        f"Tokens: {result.total_input_tokens + result.total_output_tokens} | "
        f"Cost: ${result.total_cost_usd:.6f}"
    )
    export_trace_json(result, trace_output)
    console.print(f"Trace written to {trace_output}")
    provider = tracing_provider(get_settings())
    if provider == "langsmith":
        run_id = log_langsmith_run(get_settings(), result)
        if run_id:
            console.print(f"LangSmith trace uploaded: run_id={run_id}")
        else:
            console.print("LangSmith key set but upload failed; check logs.")


@app.command()
def benchmark(
    config_path: Annotated[
        Path, typer.Option("--config", help="YAML config with benchmark queries")
    ] = DEFAULT_CONFIG,
    output: Annotated[Path, typer.Option("--output", help="Report output path")] = Path(
        "reports/benchmark_report.md"
    ),
) -> None:
    """Benchmark single-agent vs multi-agent and write a markdown report."""
    _init()
    queries = _load_benchmark_queries(config_path)
    if not queries:
        console.print("No benchmark queries found in config.", style="red")
        raise typer.Exit(code=1)

    metrics: list[BenchmarkMetrics] = []
    table = Table(title="Benchmark Runs")
    table.add_column("Query")
    table.add_column("Mode")
    table.add_column("Latency (s)")
    table.add_column("Cost (USD)")
    table.add_column("Quality")
    table.add_column("Citation cov.")
    table.add_column("Failures")

    for query in queries:
        runners = (("baseline", _baseline_runner), ("multi-agent", _multi_agent_runner))
        for run_name, runner in runners:
            state, metric = run_benchmark(run_name, query, runner)
            metrics.append(metric)
            cost = "" if metric.estimated_cost_usd is None else f"{metric.estimated_cost_usd:.5f}"
            quality = "" if metric.quality_score is None else f"{metric.quality_score:.1f}"
            citation = "" if metric.citation_coverage is None else f"{metric.citation_coverage:.0%}"
            table.add_row(
                query[:40],
                run_name,
                f"{metric.latency_seconds:.2f}",
                cost,
                quality,
                citation,
                "yes" if metric.failure_rate else "no",
            )

    console.print(table)
    report = render_markdown_report(metrics)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    console.print(f"Report written to {output}")


if __name__ == "__main__":
    app()
