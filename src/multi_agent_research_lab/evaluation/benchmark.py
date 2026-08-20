"""Benchmark for single-agent vs multi-agent pipelines."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import AgentName, BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality, citation coverage, and failure rate."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=state.total_cost_usd,
        quality_score=quality_score(state),
        citation_coverage=citation_coverage(state),
        failure_rate=1.0 if state.errors else 0.0,
        notes=build_notes(state),
    )
    return state, metrics


def quality_score(state: ResearchState) -> float:
    """Heuristic 0-10 quality: answer presence, depth, sourcing, citations."""
    score = 0.0
    answer = state.final_answer or ""
    if answer and len(answer.split()) >= 30:
        score += 4.0
    elif answer:
        score += 2.0
    if state.sources:
        score += 2.0
    if re.search(r"\[[^\]]+\]", answer):
        score += 3.0
    if state.research_notes and state.analysis_notes:
        score += 1.0
    return min(score, 10.0)


def citation_coverage(state: ResearchState) -> float:
    """Fraction of known sources actually cited in the final answer."""
    known = {
        str(s.metadata.get("source_id") or f"s{i}") for i, s in enumerate(state.sources, start=1)
    }
    if not known:
        return 0.0
    cited = set(re.findall(r"\[([^\]]+)\]", state.final_answer or ""))
    return len(cited & known) / len(known)


def build_notes(state: ResearchState) -> str:
    parts = [
        f"provider={state.llm_provider}",
        f"tokens={state.total_input_tokens}+{state.total_output_tokens}",
    ]
    agents = {r.agent for r in state.agent_results}
    parts.append("agents=" + ",".join(sorted(a.value for a in agents)))
    if state.errors:
        parts.append("errors=" + ";".join(state.errors))
    return " | ".join(parts)


def agent_names(state: ResearchState) -> list[AgentName]:
    return [r.agent for r in state.agent_results]
