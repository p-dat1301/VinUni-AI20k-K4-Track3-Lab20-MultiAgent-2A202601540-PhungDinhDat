"""LLM client abstraction with provider selection.

Agents depend on this interface instead of importing an SDK directly. Retry,
timeout, and token accounting live here, not inside agents.
"""

from dataclasses import dataclass
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


MODEL_PRICING_USD_PER_MTOKEN: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate USD cost for a call, or None when the model is not priced."""
    pricing = MODEL_PRICING_USD_PER_MTOKEN.get(model)
    if pricing is None:
        return None
    input_rate, output_rate = pricing
    return (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)


class LLMClient:
    """Provider-agnostic LLM client."""

    provider: str = "base"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """Real OpenAI chat-completions client with retry and timeout."""

    provider = "openai"

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._model = settings.openai_model
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
        input_tokens = resp.usage.prompt_tokens if resp.usage else None
        output_tokens = resp.usage.completion_tokens if resp.usage else None
        cost = (
            estimate_cost(self._model, input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


class OfflineLLMClient(LLMClient):
    """Deterministic offline fallback that works without any API key."""

    provider = "offline-mock"

    def __init__(self, settings: Settings, corpus_dir: Path | None = None) -> None:
        self._model = settings.openai_model
        self._corpus_dir = corpus_dir

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        context = self._extract_context(user_prompt)
        role = self._detect_role(system_prompt)
        if role == "researcher":
            content = self._research_notes(context)
        elif role == "analyst":
            content = self._analysis_notes(context)
        elif role == "writer":
            content = self._final_answer(context)
        else:
            content = self._generic(context)
        input_tokens = max(1, len(system_prompt + user_prompt) // 4)
        output_tokens = max(1, len(content) // 4)
        cost = estimate_cost(self._model, input_tokens, output_tokens)
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _extract_context(self, user_prompt: str) -> str:
        marker = "CONTEXT:"
        if marker in user_prompt:
            return user_prompt.split(marker, 1)[1].strip()
        return ""

    def _detect_role(self, system_prompt: str) -> str:
        lowered = system_prompt.lower()
        for role in ("researcher", "analyst", "writer"):
            if role in lowered:
                return role
        return "generic"

    def _research_notes(self, context: str) -> str:
        if not context:
            return "No search context available. No sources were retrieved."
        lines = ["Research notes compiled from retrieved sources (offline corpus):", ""]
        for source_id, title, snippet in self._iter_source_blocks(context):
            lines.append(f"- {title} [{source_id}]: {snippet}")
        return "\n".join(lines)

    def _analysis_notes(self, context: str) -> str:
        blocks = self._iter_source_blocks(context)
        if not blocks:
            return "Analysis: insufficient evidence to extract claims."
        lines = ["Analysis notes:", ""]
        for idx, (source_id, title, snippet) in enumerate(blocks, start=1):
            claim = snippet.split(".")[0] if snippet else title
            lines.append(
                f"{idx}. Claim: {claim}. Evidence: {title} [{source_id}]. "
                "Weakness: single-source evidence."
            )
        lines.extend(
            [
                "",
                "Viewpoint comparison: sources agree on core facts; synthetic documents "
                "are flagged as benchmark material.",
                "Weak evidence flagged: claims backed by only one source need cross-checking.",
            ]
        )
        return "\n".join(lines)

    def _final_answer(self, context: str) -> str:
        blocks = self._iter_source_blocks(context)
        if not blocks:
            return "No sources were retrieved, so no evidence-backed answer can be produced."
        lines = ["Summary", ""]
        for idx, (source_id, title, snippet) in enumerate(blocks, start=1):
            first = snippet.split(".")[0] if snippet else title
            lines.append(f"{idx}. {first} [{source_id}].")
        lines.extend(["", "Conclusions", ""])
        cited = ", ".join(f"[{b[0]}]" for b in blocks[:3])
        lines.append(
            f"The evidence above ({cited}) supports the main claims. "
            "Synthetic benchmark documents were not presented as real publications."
        )
        return "\n".join(lines)

    def _generic(self, context: str) -> str:
        if context:
            return f"Summary of retrieved context: {context[:400]}"
        return "No response could be generated without context or an API key."

    @staticmethod
    def _iter_source_blocks(context: str) -> list[tuple[str, str, str]]:
        blocks: list[tuple[str, str, str]] = []
        for line in context.splitlines():
            line = line.strip()
            if not line.startswith("- [") or "] " not in line:
                continue
            rest = line[2:]
            source_id, remainder = rest.split("]", 1)
            source_id = source_id.strip(" [")
            title, _, snippet = remainder.strip().partition(":")
            blocks.append((source_id, title.strip(), snippet.strip()))
        return blocks


def build_llm_client(settings: Settings, corpus_dir: Path | None = None) -> LLMClient:
    """Pick the real OpenAI client when a key exists, else the offline mock."""
    if settings.openai_api_key:
        return OpenAILLMClient(settings)
    return OfflineLLMClient(settings, corpus_dir)
