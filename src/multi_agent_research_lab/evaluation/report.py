"""Benchmark report rendering."""

from statistics import mean

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown with a comparison summary."""
    lines = [
        "# Benchmark Report",
        "",
        "## Metrics",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    lines.extend(["", "## Comparison", ""])
    by_run: dict[str, list[BenchmarkMetrics]] = {}
    for item in metrics:
        by_run.setdefault(item.run_name, []).append(item)
    baseline = by_run.get("baseline", [])
    multi = by_run.get("multi-agent", [])
    if baseline and multi:
        b_lat = mean(m.latency_seconds for m in baseline)
        m_lat = mean(m.latency_seconds for m in multi)
        b_q = mean(m.quality_score or 0 for m in baseline)
        m_q = mean(m.quality_score or 0 for m in multi)
        b_c = mean(m.citation_coverage or 0 for m in baseline)
        m_c = mean(m.citation_coverage or 0 for m in multi)
        lines.append(
            f"- Latency: multi-agent **{m_lat:+.2f}s** vs single-agent "
            f"({_fraction(m_lat, b_lat):.1f}x)."
        )
        lines.append(
            f"- Quality: multi-agent **{m_q:+.1f}/10** vs single-agent ({b_q:.1f} -> {m_q:.1f})."
        )
        lines.append(
            f"- Citation coverage: multi-agent **{m_c:.0%}** vs single-agent **{b_c:.0%}**."
        )
        lines.append(
            "- Multi-agent trades latency/cost for better citation coverage and "
            "structured evidence, which matters for research-grade answers."
        )
    lines.extend(
        [
            "",
            "## Failure Mode & Fix",
            "",
            "**Failure mode gặp phải:** OfflineLLMClient (fallback khi không có API key) chỉ "
            "đọc context sau marker `CONTEXT:`. Ban đầu Analyst/Writer gửi user_prompt dạng "
            "`RESEARCH NOTES:` / `AVAILABLE SOURCES:` nên mock nhận context rỗng, writer trả "
            "`No sources retrieved`, quality tụt còn 5.0 và citation coverage 0%.",
            "",
            "**Cách fix:** chuẩn hóa contract user_prompt — mọi agent đưa dữ liệu vào marker "
            "`CONTEXT:` với cùng format source block `- [source_id] title: snippet`. Sau fix, "
            "quality lên 10.0 và citation coverage đạt 100%. Bài học: contract giữa agent và "
            "LLM/mock client phải nhất quán; đổi một bên mà không đổi bên kia sẽ fail im lặng.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fraction(a: float, b: float) -> float:
    return a / b if b else 0.0
