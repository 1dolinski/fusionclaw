"""Tests for InternalRegistry — config loading, decorator, search, select."""

import asyncio
import json
import tempfile
from pathlib import Path

from fusionclaw.claw import BaseClaw
from fusionclaw.models import Fact, StateObject
from fusionclaw.registry import (
    InternalRegistry,
    _StubClaw,
    get_default_registry,
    register,
    reset_default_registry,
)


class AlphaClaw(BaseClaw):
    claw_id = "alpha"
    description = "Alpha specialist for testing"

    async def run(self, input: str) -> StateObject:
        return StateObject(claw_id=self.claw_id, summary=f"Alpha: {input}")


class BetaClaw(BaseClaw):
    claw_id = "beta"
    description = "Beta analyst for pricing"

    async def run(self, input: str) -> StateObject:
        return StateObject(claw_id=self.claw_id, summary=f"Beta: {input}")


# --- Basic Registry Operations ---


def test_add_and_get():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    assert reg.get("alpha") is not None
    assert reg.get("alpha").claw_id == "alpha"


def test_add_and_list():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    reg.add(BetaClaw())
    assert len(reg) == 2
    assert len(reg.list()) == 2


def test_list_ids():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    reg.add(BetaClaw())
    ids = reg.list_ids()
    assert "alpha" in ids
    assert "beta" in ids


def test_remove():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    reg.remove("alpha")
    assert reg.get("alpha") is None
    assert len(reg) == 0


def test_remove_nonexistent():
    reg = InternalRegistry()
    reg.remove("nope")  # should not raise


def test_contains():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    assert "alpha" in reg
    assert "beta" not in reg


# --- Search ---


def test_search_by_id():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    reg.add(BetaClaw())
    results = reg.search("alpha")
    assert len(results) == 1
    assert results[0].claw_id == "alpha"


def test_search_by_description():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    reg.add(BetaClaw())
    results = reg.search("pricing")
    assert len(results) == 1
    assert results[0].claw_id == "beta"


def test_search_case_insensitive():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    results = reg.search("ALPHA")
    assert len(results) == 1


def test_search_no_match():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    results = reg.search("zzzzzz")
    assert len(results) == 0


# --- Select ---


def test_select():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    reg.add(BetaClaw())
    selected = reg.select(["alpha", "beta"])
    assert len(selected) == 2
    assert selected[0].claw_id == "alpha"
    assert selected[1].claw_id == "beta"


def test_select_missing_raises():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    try:
        reg.select(["alpha", "nonexistent"])
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "nonexistent" in str(e)


# --- JSON Config ---


def test_from_json_with_inline_claws():
    config = {
        "claws": [
            {
                "claw_id": "stub1",
                "description": "A stub claw",
                "summary": "stub summary",
                "facts": [{"key": "k", "value": "v"}],
            },
            {
                "claw_id": "stub2",
                "description": "Another stub",
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        f.flush()
        reg = InternalRegistry.from_json(f.name)

    assert len(reg) == 2
    assert reg.get("stub1") is not None
    assert isinstance(reg.get("stub1"), _StubClaw)


def test_from_dict():
    config = {
        "claws": [
            {"claw_id": "inline", "description": "Inline claw"},
        ]
    }
    reg = InternalRegistry.from_dict(config)
    assert len(reg) == 1
    assert "inline" in reg


def test_stub_claw_run():
    stub = _StubClaw(
        claw_id="test_stub",
        description="Test stub",
        static_summary="A test",
        static_facts=[{"key": "color", "value": "blue"}],
    )
    result = asyncio.get_event_loop().run_until_complete(stub.run("anything"))
    assert result.claw_id == "test_stub"
    assert result.summary == "A test"
    assert len(result.key_facts) == 1
    assert result.key_facts[0].key == "color"


# --- Decorator ---


def test_register_decorator():
    reset_default_registry()

    @register()
    class DecoratedClaw(BaseClaw):
        claw_id = "decorated"
        description = "Registered via decorator"

        async def run(self, input: str) -> StateObject:
            return StateObject(claw_id=self.claw_id, summary="dec")

    reg = get_default_registry()
    assert "decorated" in reg
    assert reg.get("decorated").claw_id == "decorated"
    reset_default_registry()


def test_register_decorator_custom_registry():
    custom = InternalRegistry()

    @register(registry=custom)
    class CustomClaw(BaseClaw):
        claw_id = "custom"
        description = "In custom registry"

        async def run(self, input: str) -> StateObject:
            return StateObject(claw_id=self.claw_id, summary="c")

    assert "custom" in custom
    assert len(custom) == 1


# --- Serialization ---


def test_to_dict():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    reg.add(BetaClaw())
    d = reg.to_dict()
    assert len(d["claws"]) == 2
    ids = [c["claw_id"] for c in d["claws"]]
    assert "alpha" in ids
    assert "beta" in ids


def test_to_json():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    j = reg.to_json()
    data = json.loads(j)
    assert len(data["claws"]) == 1


def test_to_json_file():
    reg = InternalRegistry()
    reg.add(AlphaClaw())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        reg.to_json(f.name)
    content = Path(f.name).read_text()
    data = json.loads(content)
    assert data["claws"][0]["claw_id"] == "alpha"
