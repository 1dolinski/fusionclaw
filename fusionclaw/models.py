from __future__ import annotations

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A single verifiable data point extracted by a claw."""

    key: str
    value: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class StateObject(BaseModel):
    """Structured context exported by a claw after doing its work.

    This is the core unit of fusion — instead of chatting,
    claws export these and the fuser merges them directly.
    """

    claw_id: str
    summary: str
    key_facts: list[Fact] = Field(default_factory=list)
    raw_context: str = ""
    token_count: int = 0


class ContextBlock(BaseModel):
    """A single block within a fused context window."""

    source_claw_id: str
    content: str
    is_compressed: bool = False
    original_tokens: int = 0
    final_tokens: int = 0


class FusedContext(BaseModel):
    """The merged context window ready for synthesis."""

    blocks: list[ContextBlock]
    total_tokens: int = 0
    compression_applied: bool = False


class SynthesisResult(BaseModel):
    """Final output from the orchestrator."""

    answer: str
    fused_context: FusedContext
    model: str
    usage: TokenUsage


class TokenUsage(BaseModel):
    """Token usage stats from LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
