"""Tests for BaseClaw."""

import asyncio

from fusionclaw.claw import BaseClaw
from fusionclaw.models import Fact, StateObject


class DummyClaw(BaseClaw):
    claw_id = "dummy"
    description = "A test claw"

    async def run(self, input: str) -> StateObject:
        return StateObject(
            claw_id=self.claw_id,
            summary=f"Analyzed: {input}",
            key_facts=[Fact(key="input", value=input)],
            raw_context=f"Full analysis of {input}",
            token_count=10,
        )


def test_claw_subclass():
    claw = DummyClaw()
    assert claw.claw_id == "dummy"
    assert claw.description == "A test claw"


def test_claw_run():
    claw = DummyClaw()
    result = asyncio.get_event_loop().run_until_complete(claw.run("test input"))
    assert result.claw_id == "dummy"
    assert "test input" in result.summary
    assert len(result.key_facts) == 1


def test_multiple_claws():
    """Multiple claws can coexist with different IDs."""

    class ClawA(BaseClaw):
        claw_id = "a"
        description = "Claw A"

        async def run(self, input: str) -> StateObject:
            return StateObject(claw_id=self.claw_id, summary="A")

    class ClawB(BaseClaw):
        claw_id = "b"
        description = "Claw B"

        async def run(self, input: str) -> StateObject:
            return StateObject(claw_id=self.claw_id, summary="B")

    a, b = ClawA(), ClawB()
    assert a.claw_id != b.claw_id
