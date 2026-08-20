"""Search client abstraction for ResearcherAgent.

Two backends: a real Tavily client when TAVILY_API_KEY is set, and an offline
corpus search over `ai_agent_offline_research_corpus_v2/` that works with no
network or keys so the lab can be exercised end-to-end.
"""

import json
import re
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import SourceDocument

DEFAULT_CORPUS_DIR = Path("ai_agent_offline_research_corpus_v2/topics")


class SearchClient:
    """Provider-agnostic search client."""

    provider: str = "base"

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        raise NotImplementedError


class TavilySearchClient(SearchClient):
    """Real Tavily search client."""

    provider = "tavily"

    def __init__(self, api_key: str, timeout_seconds: int = 30) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        import urllib.request

        payload = json.dumps(
            {
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self._timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        docs = []
        for item in data.get("results", []):
            docs.append(
                SourceDocument(
                    title=item.get("title", "Untitled"),
                    url=item.get("url"),
                    snippet=item.get("content", ""),
                    metadata={"score": item.get("score"), "provider": "tavily"},
                )
            )
        return docs


class OfflineCorpusSearchClient(SearchClient):
    """Keyword search over the bundled offline research corpus."""

    provider = "offline-corpus"

    def __init__(self, corpus_dir: Path = DEFAULT_CORPUS_DIR) -> None:
        self._corpus_dir = corpus_dir
        self._topics: list[dict[str, Any]] | None = None

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        topic = self._best_topic(query)
        if topic is None:
            return []
        docs = self._rank_sources(topic, query, max_results)
        return docs

    def _load_topics(self) -> list[dict[str, Any]]:
        if self._topics is not None:
            return self._topics
        topics: list[dict[str, Any]] = []
        if self._corpus_dir.is_dir():
            for path in sorted(self._corpus_dir.glob("*.json")):
                try:
                    with path.open(encoding="utf-8") as fh:
                        data = json.load(fh)
                    topics.append({"path": path, "data": data})
                except (json.JSONDecodeError, OSError):
                    continue
        self._topics = topics
        return topics

    def _best_topic(self, query: str) -> dict[str, Any] | None:
        query_terms = set(_tokenize(query))
        best: tuple[float, dict[str, Any] | None] = (0.0, None)
        for topic in self._load_topics():
            data = topic["data"]
            topic_info = data.get("topic", {})
            haystack = " ".join(
                [
                    str(topic_info.get("name", "")),
                    " ".join(topic_info.get("tags", [])),
                    str(topic_info.get("research_question", "")),
                ]
            )
            score = _overlap(query_terms, set(_tokenize(haystack)))
            if score > best[0]:
                best = (score, topic)
        return best[1]

    def _rank_sources(
        self, topic: dict[str, Any], query: str, max_results: int
    ) -> list[SourceDocument]:
        query_terms = set(_tokenize(query))
        knowledge = topic["data"].get("knowledge_base", {})
        source_docs = knowledge.get("source_documents", [])
        ranked = sorted(
            source_docs,
            key=lambda doc: _overlap(
                query_terms,
                set(_tokenize(f"{doc.get('title', '')} {doc.get('key_takeaways', '')}")),
            ),
            reverse=True,
        )
        results = []
        for doc in ranked[:max_results]:
            takeaways = doc.get("key_takeaways", [])
            if isinstance(takeaways, list):
                snippet = " ".join(str(t) for t in takeaways)
            else:
                snippet = str(takeaways)
            if not snippet:
                snippet = str(doc.get("full_text", ""))[:400]
            results.append(
                SourceDocument(
                    title=doc.get("title", "Untitled"),
                    url=doc.get("provenance_url"),
                    snippet=snippet,
                    metadata={
                        "source_id": doc.get("document_id") or doc.get("citation_label"),
                        "is_synthetic": doc.get("is_synthetic", False),
                        "topic_file": str(topic["path"]),
                        "provider": "offline-corpus",
                    },
                )
            )
        return results


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), 1)


def build_search_client(settings: Settings, corpus_dir: Path | None = None) -> SearchClient:
    """Pick Tavily when a key exists, otherwise fall back to the offline corpus."""
    if settings.tavily_api_key:
        return TavilySearchClient(settings.tavily_api_key, settings.timeout_seconds)
    return OfflineCorpusSearchClient(corpus_dir or DEFAULT_CORPUS_DIR)
