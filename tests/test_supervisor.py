"""Routing policy tests for the supervisor agent."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state_with(**overrides: object) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def test_routes_to_researcher_first() -> None:
    state = _state_with()
    SupervisorAgent().run(state)
    assert state.route_history == ["researcher"]
    assert state.iteration == 1


def test_routes_to_analyst_after_research() -> None:
    state = _state_with(research_notes="notes")
    SupervisorAgent().run(state)
    assert state.route_history == ["analyst"]


def test_routes_to_writer_after_analysis() -> None:
    state = _state_with(research_notes="notes", analysis_notes="analysis")
    SupervisorAgent().run(state)
    assert state.route_history == ["writer"]


def test_routes_to_critic_before_done() -> None:
    state = _state_with(research_notes="notes", analysis_notes="analysis", final_answer="answer")
    SupervisorAgent().run(state)
    assert state.route_history == ["critic"]


def test_done_after_critic() -> None:
    state = _state_with(research_notes="notes", analysis_notes="analysis", final_answer="answer")
    state.agent_results.append(AgentResult(agent=AgentName.CRITIC, content="ok"))
    SupervisorAgent().run(state)
    assert state.route_history == ["done"]


def test_max_iterations_enforced() -> None:
    state = _state_with(research_notes="notes", analysis_notes="analysis", final_answer="answer")
    agent = SupervisorAgent(max_iterations=0)
    agent.run(state)
    assert state.route_history == ["done"]
