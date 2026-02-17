"""Tests for ContextFuser."""

from fusionclaw.fuser import ContextFuser, count_tokens
from fusionclaw.models import Fact, StateObject


def _make_state(claw_id: str, context_size: int = 50) -> StateObject:
    """Helper to make a StateObject with controllable context size."""
    raw = "word " * context_size
    return StateObject(
        claw_id=claw_id,
        summary=f"{claw_id} summary",
        key_facts=[
            Fact(key="fact1", value="value1"),
            Fact(key="fact2", value="value2"),
        ],
        raw_context=raw,
        token_count=count_tokens(raw),
    )


def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


def test_full_merge_under_budget():
    fuser = ContextFuser(token_budget=100_000)
    states = [_make_state("a"), _make_state("b")]
    result = fuser.fuse(states)
    assert not result.compression_applied
    assert len(result.blocks) == 2
    assert all(not b.is_compressed for b in result.blocks)


def test_compressed_merge_over_budget():
    fuser = ContextFuser(token_budget=50)
    states = [_make_state("a", context_size=200), _make_state("b", context_size=200)]
    result = fuser.fuse(states, token_budget=50)
    assert result.compression_applied
    assert any(b.is_compressed for b in result.blocks)


def test_priority_keeps_important_claw_full():
    fuser = ContextFuser()
    states = [_make_state("important", 200), _make_state("nice_to_have", 200)]

    # Budget fits one full + one compressed
    full_tokens = count_tokens(fuser._format_full_block(states[0]))
    compressed_tokens = count_tokens(fuser._format_compressed_block(states[1]))
    budget = full_tokens + compressed_tokens + 10

    result = fuser.fuse(states, token_budget=budget, priorities={"important": 10, "nice_to_have": 1})

    important_block = next(b for b in result.blocks if b.source_claw_id == "important")
    nice_block = next(b for b in result.blocks if b.source_claw_id == "nice_to_have")
    assert not important_block.is_compressed
    assert nice_block.is_compressed


def test_build_prompt_contains_query():
    fuser = ContextFuser()
    states = [_make_state("a")]
    fused = fuser.fuse(states)
    prompt = fuser.build_prompt(fused, "What is the threat?")
    assert "What is the threat?" in prompt
    assert "FUSED_CONTEXT" in prompt
    assert "CONTEXT_BLOCK" in prompt


def test_build_prompt_tags_compression():
    fuser = ContextFuser(token_budget=30)
    states = [_make_state("a", 200)]
    fused = fuser.fuse(states, token_budget=30)
    prompt = fuser.build_prompt(fused, "query")
    assert "COMPRESSED" in prompt


def test_empty_states():
    fuser = ContextFuser()
    result = fuser.fuse([])
    assert len(result.blocks) == 0
    assert result.total_tokens == 0
    assert not result.compression_applied


def test_facts_only_fallback():
    """When even compressed doesn't fit, falls back to facts only."""
    fuser = ContextFuser()
    states = [_make_state("a", 500), _make_state("b", 500)]

    # Very tight budget — only facts should fit
    result = fuser.fuse(states, token_budget=25)
    assert result.compression_applied
    # Should have at least some blocks
    assert len(result.blocks) >= 1
