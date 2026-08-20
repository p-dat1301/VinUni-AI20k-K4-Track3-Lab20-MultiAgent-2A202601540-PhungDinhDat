"""Tests for the offline LLM fallback and cost estimation."""

from pathlib import Path

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.services.llm_client import (
    OfflineLLMClient,
    build_llm_client,
    estimate_cost,
)

CORPUS = Path("ai_agent_offline_research_corpus_v2/topics")


def test_estimate_cost_known_model() -> None:
    cost = estimate_cost("gpt-4o-mini", 1_000_000, 0)
    assert cost == 0.15


def test_estimate_cost_unknown_model_is_none() -> None:
    assert estimate_cost("unknown-model", 100, 100) is None


def test_offline_mock_researcher_role() -> None:
    client = OfflineLLMClient(Settings(), CORPUS)
    response = client.complete(
        "You are the Researcher agent.", "QUERY: x\nCONTEXT:\n- [s1] Title: Snippet text here."
    )
    assert "[s1]" in response.content
    assert response.input_tokens and response.output_tokens


def test_build_llm_client_without_key_returns_offline() -> None:
    client = build_llm_client(Settings(openai_api_key=None), CORPUS)
    assert client.provider == "offline-mock"
