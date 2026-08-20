"""Supervisor / router agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker runs next and when to stop."""

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        settings = get_settings()
        self.max_iterations = settings.max_iterations if max_iterations is None else max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Deterministic pipeline: researcher -> analyst -> writer -> critic -> done.
        Hard stop when the iteration budget is exhausted.
        """

        next_route = self._decide(state)
        state.record_route(next_route)
        state.add_trace_event("supervisor", {"next": next_route, "iteration": state.iteration})
        return state

    def _decide(self, state: ResearchState) -> str:
        if state.iteration >= self.max_iterations:
            return "done"
        if state.research_notes is None:
            return "researcher"
        if state.analysis_notes is None:
            return "analyst"
        if state.final_answer is None:
            return "writer"
        if not self._critic_done(state):
            return "critic"
        return "done"

    @staticmethod
    def _critic_done(state: ResearchState) -> bool:
        return any(result.agent == AgentName.CRITIC for result in state.agent_results)
