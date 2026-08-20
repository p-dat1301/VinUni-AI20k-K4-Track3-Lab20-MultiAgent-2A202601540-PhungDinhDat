"""Tracing hooks.

Provider-agnostic: JSON traces by default; LangSmith is used when
LANGSMITH_API_KEY is configured.
"""

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context that records wall-clock duration."""
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started


def tracing_provider(settings: Settings) -> str:
    """Return the active tracing provider name based on configured keys."""
    if settings.langsmith_api_key:
        return "langsmith"
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        return "langfuse"
    return "json"


def export_trace_json(state: ResearchState, path: Path) -> Path:
    """Persist the workflow trace as JSON so it can be shared or replayed."""
    payload = {
        "query": state.request.query,
        "route_history": state.route_history,
        "trace": state.trace,
        "errors": state.errors,
        "final_answer": state.final_answer,
        "sources": [s.model_dump() for s in state.sources],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def log_langsmith_run(settings: Settings, state: ResearchState) -> str | None:
    """Upload a trace run to LangSmith when a key is configured.

    No-op (returns None) without a key or if the SDK is unavailable, so the
    workflow never fails because of tracing.
    """

    if not settings.langsmith_api_key:
        return None
    try:
        from langsmith import Client

        client = Client(api_key=settings.langsmith_api_key)
        run_id = uuid.uuid4()
        started = datetime.now(UTC)
        client.create_run(
            id=run_id,
            name="multi-agent-research",
            run_type="chain",
            inputs={"query": state.request.query},
            outputs={"route_history": state.route_history, "final_answer": state.final_answer},
            start_time=started,
            end_time=started,
        )
        client.update_run(run_id, outputs={"trace": state.trace})
        url = client.get_run_url(run=run_id)
        return str(url) if url else str(run_id)
    except Exception:  # noqa: BLE001 - tracing is best-effort
        logger.warning("LangSmith tracing failed; continuing without it.", exc_info=True)
        return None
