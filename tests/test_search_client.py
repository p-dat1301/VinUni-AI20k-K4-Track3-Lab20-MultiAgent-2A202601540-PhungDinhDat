"""Tests for the offline corpus search client."""

from pathlib import Path

from multi_agent_research_lab.services.search_client import OfflineCorpusSearchClient

CORPUS = Path("ai_agent_offline_research_corpus_v2/topics")


def test_offline_search_returns_sources() -> None:
    client = OfflineCorpusSearchClient(CORPUS)
    docs = client.search("multi-agent architectures for research tasks", max_results=5)
    assert docs
    assert all(doc.title for doc in docs)
    assert all(doc.snippet for doc in docs)
    assert all(doc.metadata.get("source_id") for doc in docs)


def test_offline_search_returns_empty_for_garbage_query() -> None:
    client = OfflineCorpusSearchClient(CORPUS)
    docs = client.search("zzzzzqqqqq", max_results=5)
    assert docs == []
