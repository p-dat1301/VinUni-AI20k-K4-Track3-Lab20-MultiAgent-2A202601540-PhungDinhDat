"""Critic agent: verify citation discipline of the final answer."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Fact-checks citation coverage and flags hallucinated references."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Append a critique with citation coverage and hallucination findings."""

        answer = state.final_answer or ""
        known_ids = {
            str(s.metadata.get("source_id") or f"s{i}")
            for i, s in enumerate(state.sources, start=1)
        }
        cited = set(re.findall(r"\[([^\]]+)\]", answer))
        valid = cited & known_ids
        hallucinated = cited - known_ids
        coverage = len(valid) / len(known_ids) if known_ids else 0.0
        findings = (
            f"Citation coverage: {coverage:.0%} ({len(valid)}/{len(known_ids)} sources cited). "
            f"Hallucinated references: {sorted(hallucinated) if hallucinated else 'none'}."
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=findings,
                metadata={"citation_coverage": coverage, "hallucinated": sorted(hallucinated)},
            )
        )
        state.add_trace_event(
            "critic", {"coverage": coverage, "hallucinated": sorted(hallucinated)}
        )
        return state
