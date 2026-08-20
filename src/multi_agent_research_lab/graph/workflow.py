"""LangGraph workflow for the multi-agent research system.

Orchestration lives here; agent internals live in `agents/`.
"""

from collections.abc import Callable
from typing import Any, TypedDict, cast

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.state import ResearchState


class GraphState(TypedDict, total=False):
    request: dict[str, Any]
    iteration: int
    route_history: list[str]
    sources: list[dict[str, Any]]
    research_notes: str | None
    analysis_notes: str | None
    final_answer: str | None
    agent_results: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    errors: list[str]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    llm_provider: str
    route: str


def _to_state(data: GraphState) -> ResearchState:
    raw = cast(dict[str, Any], data)
    return ResearchState(**{k: v for k, v in raw.items() if k != "route"})


def _make_worker_node(
    agent: WriterAgent | ResearcherAgent | AnalystAgent | CriticAgent,
) -> Callable[..., GraphState]:
    def node(state: GraphState) -> GraphState:
        return cast(GraphState, agent.run(_to_state(state)).model_dump())

    return node


def _make_supervisor_node(agent: SupervisorAgent) -> Callable[..., GraphState]:
    def node(state: GraphState) -> GraphState:
        research_state = _to_state(state)
        agent.run(research_state)
        dumped = research_state.model_dump()
        dumped["route"] = research_state.route_history[-1]
        return cast(GraphState, dumped)

    return node


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self) -> None:
        self._supervisor = SupervisorAgent()
        self._researcher = ResearcherAgent()
        self._analyst = AnalystAgent()
        self._writer = WriterAgent()
        self._critic = CriticAgent()

    def build(self) -> Any:
        """Create a compiled LangGraph graph."""
        from langgraph.graph import END, StateGraph

        graph = StateGraph(GraphState)

        graph.add_node("supervisor", _make_supervisor_node(self._supervisor))
        graph.add_node("researcher", _make_worker_node(self._researcher))
        graph.add_node("analyst", _make_worker_node(self._analyst))
        graph.add_node("writer", _make_worker_node(self._writer))
        graph.add_node("critic", _make_worker_node(self._critic))

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["route"],
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            },
        )
        for worker in ("researcher", "analyst", "writer", "critic"):
            graph.add_edge(worker, "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final state."""
        try:
            compiled = self.build()
        except ImportError:
            return self._manual_run(state)
        result = compiled.invoke(state.model_dump())
        return _to_state(result)

    def _manual_run(self, state: ResearchState) -> ResearchState:
        """Fallback loop when LangGraph is not installed."""
        while state.iteration < self._supervisor.max_iterations:
            self._supervisor.run(state)
            route = state.route_history[-1]
            if route == "done":
                break
            if route == "researcher":
                self._researcher.run(state)
            elif route == "analyst":
                self._analyst.run(state)
            elif route == "writer":
                self._writer.run(state)
            elif route == "critic":
                self._critic.run(state)
        return state
