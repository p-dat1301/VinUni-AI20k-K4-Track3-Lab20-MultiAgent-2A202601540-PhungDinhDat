"""Analyst agent: turn research notes into structured insights."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, build_llm_client

ANALYST_SYSTEM_PROMPT = (
    "You are the Analyst agent in a multi-agent research system. "
    "Extract key claims from the research notes, compare viewpoints, "
    "and flag weak evidence. Keep analysis structured and traceable to sources."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        settings = get_settings()
        self._llm = llm_client or build_llm_client(settings, settings.offline_corpus_dir)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        notes = state.research_notes or "No research notes available."
        try:
            response = self._llm.complete(ANALYST_SYSTEM_PROMPT, f"CONTEXT:\n{notes}")
            state.analysis_notes = response.content
            self.record_usage(state, self._llm, response)
            state.agent_results.append(
                AgentResult(agent=AgentName.ANALYST, content=response.content)
            )
        except Exception as exc:  # noqa: BLE001 - fallback must never crash the graph
            state.errors.append(f"analyst: {exc}")
            state.analysis_notes = state.analysis_notes or (
                "Fallback: analysis skipped after analyst failure."
            )
        state.add_trace_event("analyst", {"has_analysis": bool(state.analysis_notes)})
        return state
