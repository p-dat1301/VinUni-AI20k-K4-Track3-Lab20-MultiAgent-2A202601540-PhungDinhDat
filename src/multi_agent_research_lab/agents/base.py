"""Base agent contract.

The concrete agent classes intentionally contain TODOs. The goal is to force students
to make design decisions instead of receiving a finished implementation.
"""

from abc import ABC, abstractmethod

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse


class BaseAgent(ABC):
    """Minimal interface every agent must implement."""

    name: str

    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        """Read and update shared state, then return it."""

    @staticmethod
    def record_usage(state: ResearchState, client: LLMClient, response: LLMResponse) -> None:
        """Accumulate token/cost accounting from an LLM call into shared state."""
        state.llm_provider = client.provider
        state.record_usage(
            input_tokens=response.input_tokens or 0,
            output_tokens=response.output_tokens or 0,
            cost_usd=response.cost_usd or 0.0,
        )
