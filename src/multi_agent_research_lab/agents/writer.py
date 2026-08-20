"""Writer agent: produce the final evidence-backed answer."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, build_llm_client

WRITER_SYSTEM_PROMPT = (
    "You are the Writer agent in a multi-agent research system. "
    "Synthesize a clear final answer from the analysis notes. "
    "Cite supporting sources inline as [source_id] and never invent citations."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        settings = get_settings()
        self._llm = llm_client or build_llm_client(settings, settings.offline_corpus_dir)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        analysis = state.analysis_notes or "No analysis available."
        sources = "\n".join(
            f"- [{s.metadata.get('source_id') or f's{i}'}] {s.title}: {s.snippet[:300]}"
            for i, s in enumerate(state.sources, start=1)
        )
        try:
            response = self._llm.complete(
                WRITER_SYSTEM_PROMPT,
                f"ANALYSIS:\n{analysis}\nCONTEXT:\n{sources}",
            )
            state.final_answer = response.content
            self.record_usage(state, self._llm, response)
            state.agent_results.append(
                AgentResult(agent=AgentName.WRITER, content=response.content)
            )
        except Exception as exc:  # noqa: BLE001 - fallback must never crash the graph
            state.errors.append(f"writer: {exc}")
            state.final_answer = state.final_answer or (
                "Fallback: final answer could not be generated."
            )
        state.add_trace_event("writer", {"has_answer": bool(state.final_answer)})
        return state
