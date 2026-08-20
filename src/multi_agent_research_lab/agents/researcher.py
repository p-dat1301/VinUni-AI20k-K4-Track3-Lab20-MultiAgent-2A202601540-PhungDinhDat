"""Researcher agent: retrieve sources and produce research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, build_llm_client
from multi_agent_research_lab.services.search_client import SearchClient, build_search_client

RESEARCHER_SYSTEM_PROMPT = (
    "You are the Researcher agent in a multi-agent research system. "
    "Retrieve relevant sources and write concise, factual research notes. "
    "Cite each source inline as [source_id]."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        settings = get_settings()
        self._llm = llm_client or build_llm_client(settings, settings.offline_corpus_dir)
        self._search = search_client or build_search_client(settings, settings.offline_corpus_dir)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        try:
            sources = self._search.search(state.request.query, state.request.max_sources)
            state.sources = sources
            context = self._format_context(sources)
            response = self._llm.complete(
                RESEARCHER_SYSTEM_PROMPT,
                f"QUERY: {state.request.query}\nCONTEXT:\n{context}",
            )
            state.research_notes = response.content
            self.record_usage(state, self._llm, response)
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=response.content,
                    metadata={"sources_retrieved": len(sources)},
                )
            )
        except Exception as exc:  # noqa: BLE001 - fallback must never crash the graph
            state.errors.append(f"researcher: {exc}")
            state.research_notes = state.research_notes or (
                "Fallback: research notes unavailable due to search/LLM failure."
            )
        state.add_trace_event(
            "researcher", {"sources": len(state.sources), "has_notes": bool(state.research_notes)}
        )
        return state

    @staticmethod
    def _format_context(sources: list[SourceDocument]) -> str:
        lines = []
        for idx, source in enumerate(sources, start=1):
            source_id = source.metadata.get("source_id") or f"s{idx}"
            lines.append(f"- [{source_id}] {source.title}: {source.snippet[:300]}")
        return "\n".join(lines) or "No sources found."
