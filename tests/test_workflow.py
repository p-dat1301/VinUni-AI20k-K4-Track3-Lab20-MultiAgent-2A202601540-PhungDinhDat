"""End-to-end workflow test using the offline mock stack (no API keys)."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_workflow_runs_end_to_end_offline() -> None:
    state = ResearchState(
        request=ResearchQuery(query="Compare single-agent and multi-agent workflows")
    )
    result = MultiAgentWorkflow().run(state)
    assert result.final_answer
    assert result.research_notes
    assert result.analysis_notes
    assert result.sources
    assert result.route_history[-1] == "done"
    expected = {"researcher", "analyst", "writer", "critic", "done"}
    assert expected.issubset(set(result.route_history))
    assert result.total_output_tokens > 0


def test_workflow_trace_recorded() -> None:
    state = ResearchState(request=ResearchQuery(query="Summarize guardrails for LLM agents"))
    result = MultiAgentWorkflow().run(state)
    names = {event["name"] for event in result.trace}
    assert {"supervisor", "researcher", "analyst", "writer", "critic"}.issubset(names)
