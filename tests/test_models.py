"""Tests for Pydantic models."""

from fusionclaw.models import (
    ContextBlock,
    Fact,
    FusedContext,
    StateObject,
    SynthesisResult,
    TokenUsage,
)


def test_fact_defaults():
    f = Fact(key="price", value="$49")
    assert f.confidence == 1.0


def test_fact_custom_confidence():
    f = Fact(key="price", value="$49", confidence=0.8)
    assert f.confidence == 0.8


def test_state_object_minimal():
    s = StateObject(claw_id="test", summary="test summary")
    assert s.claw_id == "test"
    assert s.key_facts == []
    assert s.raw_context == ""
    assert s.token_count == 0


def test_state_object_full():
    s = StateObject(
        claw_id="pricing",
        summary="Prices dropped",
        key_facts=[Fact(key="price", value="$49")],
        raw_context="Full analysis...",
        token_count=500,
    )
    assert len(s.key_facts) == 1
    assert s.token_count == 500


def test_context_block():
    b = ContextBlock(
        source_claw_id="test",
        content="some content",
        is_compressed=True,
        original_tokens=1000,
        final_tokens=200,
    )
    assert b.is_compressed
    assert b.original_tokens == 1000


def test_fused_context():
    fc = FusedContext(
        blocks=[
            ContextBlock(source_claw_id="a", content="block a"),
            ContextBlock(source_claw_id="b", content="block b"),
        ],
        total_tokens=500,
        compression_applied=False,
    )
    assert len(fc.blocks) == 2
    assert not fc.compression_applied


def test_synthesis_result():
    r = SynthesisResult(
        answer="The competitor is dangerous.",
        fused_context=FusedContext(blocks=[], total_tokens=0),
        model="gpt-4o",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    assert r.usage.total_tokens == 150
