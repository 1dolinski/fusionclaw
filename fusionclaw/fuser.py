from __future__ import annotations

import tiktoken

from .models import ContextBlock, FusedContext, StateObject


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class ContextFuser:
    """Merges multiple StateObjects into a single FusedContext.

    Two modes:
    - Full merge: all raw_context concatenated (under budget)
    - Compressed merge: high-priority claws keep raw_context,
      others fall back to summary + key_facts (over budget)
    """

    def __init__(self, token_budget: int = 120_000, model: str = "gpt-4o"):
        self.token_budget = token_budget
        self.model = model

    def fuse(
        self,
        states: list[StateObject],
        *,
        token_budget: int | None = None,
        priorities: dict[str, int] | None = None,
    ) -> FusedContext:
        """Fuse multiple state objects into a single context.

        Args:
            states: StateObjects from claws.
            token_budget: Override default budget. None uses self.token_budget.
            priorities: Optional {claw_id: priority} dict. Higher = kept at full fidelity first.
        """
        budget = token_budget or self.token_budget
        priorities = priorities or {}

        # Build full blocks first
        full_blocks = []
        for state in states:
            content = self._format_full_block(state)
            tokens = count_tokens(content, self.model)
            full_blocks.append(
                ContextBlock(
                    source_claw_id=state.claw_id,
                    content=content,
                    is_compressed=False,
                    original_tokens=tokens,
                    final_tokens=tokens,
                )
            )

        total = sum(b.final_tokens for b in full_blocks)

        # If under budget, return full merge
        if total <= budget:
            return FusedContext(
                blocks=full_blocks,
                total_tokens=total,
                compression_applied=False,
            )

        # Over budget — compress lowest-priority blocks first
        paired = list(zip(states, full_blocks))
        paired.sort(key=lambda p: priorities.get(p[0].claw_id, 0), reverse=True)

        final_blocks: list[ContextBlock] = []
        remaining_budget = budget

        for state, full_block in paired:
            if remaining_budget >= full_block.final_tokens:
                # Keep at full fidelity
                final_blocks.append(full_block)
                remaining_budget -= full_block.final_tokens
            else:
                # Compress: summary + key_facts only
                compressed_content = self._format_compressed_block(state)
                compressed_tokens = count_tokens(compressed_content, self.model)

                if remaining_budget >= compressed_tokens:
                    final_blocks.append(
                        ContextBlock(
                            source_claw_id=state.claw_id,
                            content=compressed_content,
                            is_compressed=True,
                            original_tokens=full_block.original_tokens,
                            final_tokens=compressed_tokens,
                        )
                    )
                    remaining_budget -= compressed_tokens
                else:
                    # Even compressed doesn't fit — include what we can
                    facts_only = self._format_facts_only(state)
                    facts_tokens = count_tokens(facts_only, self.model)
                    if remaining_budget >= facts_tokens:
                        final_blocks.append(
                            ContextBlock(
                                source_claw_id=state.claw_id,
                                content=facts_only,
                                is_compressed=True,
                                original_tokens=full_block.original_tokens,
                                final_tokens=facts_tokens,
                            )
                        )
                        remaining_budget -= facts_tokens

        total_final = sum(b.final_tokens for b in final_blocks)
        return FusedContext(
            blocks=final_blocks,
            total_tokens=total_final,
            compression_applied=True,
        )

    def build_prompt(self, fused: FusedContext, user_query: str) -> str:
        """Build the final prompt from fused context + user query."""
        parts = [
            "<FUSED_CONTEXT>",
        ]
        for block in fused.blocks:
            tag = "COMPRESSED" if block.is_compressed else "FULL"
            parts.append(
                f'<CONTEXT_BLOCK source="{block.source_claw_id}" fidelity="{tag}">'
            )
            parts.append(block.content)
            parts.append("</CONTEXT_BLOCK>")
        parts.append("</FUSED_CONTEXT>")
        parts.append("")
        parts.append(f"<USER_QUERY>{user_query}</USER_QUERY>")

        return "\n".join(parts)

    @staticmethod
    def _format_full_block(state: StateObject) -> str:
        """Format a state as a full-fidelity block."""
        lines = [f"Summary: {state.summary}"]
        if state.key_facts:
            lines.append("Key Facts:")
            for f in state.key_facts:
                conf = f" (confidence: {f.confidence})" if f.confidence < 1.0 else ""
                lines.append(f"  - {f.key}: {f.value}{conf}")
        if state.raw_context:
            lines.append(f"Full Context:\n{state.raw_context}")
        return "\n".join(lines)

    @staticmethod
    def _format_compressed_block(state: StateObject) -> str:
        """Format a state as summary + key_facts only (no raw_context)."""
        lines = [f"Summary: {state.summary}"]
        if state.key_facts:
            lines.append("Key Facts:")
            for f in state.key_facts:
                conf = f" (confidence: {f.confidence})" if f.confidence < 1.0 else ""
                lines.append(f"  - {f.key}: {f.value}{conf}")
        return "\n".join(lines)

    @staticmethod
    def _format_facts_only(state: StateObject) -> str:
        """Format a state as key_facts only (last resort)."""
        if not state.key_facts:
            return f"[{state.claw_id}: no facts available]"
        lines = ["Key Facts:"]
        for f in state.key_facts:
            lines.append(f"  - {f.key}: {f.value}")
        return "\n".join(lines)
