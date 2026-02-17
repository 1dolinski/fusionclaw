"""Internal claw registry — simple discovery for company claw workforces.

Two ways to register claws:
  1. Config file: InternalRegistry.from_yaml("claws.yaml") or from_json("claws.json")
  2. Decorator: @register() on your BaseClaw subclass

Designed for teams with up to ~50 claws. For global discovery across
organizations, use the ERC-8004 integration instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .claw import BaseClaw
from .models import StateObject

# Module-level default registry
_default_registry: InternalRegistry | None = None


def _get_default_registry() -> "InternalRegistry":
    global _default_registry
    if _default_registry is None:
        _default_registry = InternalRegistry()
    return _default_registry


def register(registry: "InternalRegistry | None" = None):
    """Decorator to auto-register a BaseClaw subclass.

    Usage:
        @register()
        class MyClaw(BaseClaw):
            claw_id = "my_claw"
            description = "Does something"
            async def run(self, input: str) -> StateObject: ...

    The claw class is instantiated (no-arg) and added to the registry.
    """

    def decorator(cls: type[BaseClaw]) -> type[BaseClaw]:
        target = registry if registry is not None else _get_default_registry()
        instance = cls()
        target.add(instance)
        return cls

    return decorator


class InternalRegistry:
    """Simple in-memory registry for a company's internal claw workforce.

    Holds BaseClaw instances keyed by claw_id. Supports loading from
    YAML/JSON config files and programmatic registration.
    """

    def __init__(self) -> None:
        self._claws: dict[str, BaseClaw] = {}

    def add(self, claw: BaseClaw) -> None:
        """Register a claw instance."""
        self._claws[claw.claw_id] = claw

    def remove(self, claw_id: str) -> None:
        """Remove a claw by ID."""
        self._claws.pop(claw_id, None)

    def get(self, claw_id: str) -> BaseClaw | None:
        """Get a claw by ID, or None."""
        return self._claws.get(claw_id)

    def list(self) -> list[BaseClaw]:
        """List all registered claws."""
        return list(self._claws.values())

    def list_ids(self) -> list[str]:
        """List all registered claw IDs."""
        return list(self._claws.keys())

    def search(self, query: str) -> list[BaseClaw]:
        """Search claws by substring match on claw_id or description."""
        q = query.lower()
        return [
            c
            for c in self._claws.values()
            if q in c.claw_id.lower() or q in c.description.lower()
        ]

    def select(self, ids: list[str]) -> list[BaseClaw]:
        """Select specific claws by ID for a fusion pipeline.

        Raises KeyError if any ID is not found.
        """
        result = []
        for claw_id in ids:
            claw = self._claws.get(claw_id)
            if claw is None:
                available = ", ".join(self._claws.keys())
                raise KeyError(
                    f"Claw '{claw_id}' not found. Available: {available}"
                )
            result.append(claw)
        return result

    def __len__(self) -> int:
        return len(self._claws)

    def __contains__(self, claw_id: str) -> bool:
        return claw_id in self._claws

    @classmethod
    def from_json(cls, path: str | Path) -> "InternalRegistry":
        """Load a registry from a JSON config file.

        Expected format:
        {
          "claws": [
            {
              "claw_id": "pricing_analyst",
              "description": "Analyzes competitor pricing",
              "module": "mycompany.claws.pricing",
              "class": "PricingClaw"
            }
          ]
        }
        """
        data = json.loads(Path(path).read_text())
        return cls._from_config(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InternalRegistry":
        """Load a registry from a YAML config file.

        Same format as JSON but in YAML. Requires PyYAML.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML required for YAML config. Install: pip install pyyaml"
            )
        data = yaml.safe_load(Path(path).read_text())
        return cls._from_config(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InternalRegistry":
        """Load a registry from a dict (same schema as JSON config)."""
        return cls._from_config(data)

    @classmethod
    def _from_config(cls, data: dict[str, Any]) -> "InternalRegistry":
        """Parse config dict and import+instantiate claw classes."""
        import importlib

        registry = cls()
        for entry in data.get("claws", []):
            module_path = entry.get("module", "")
            class_name = entry.get("class", "")

            if module_path and class_name:
                mod = importlib.import_module(module_path)
                klass = getattr(mod, class_name)
                instance = klass()
            else:
                # Inline definition — create a stub claw
                instance = _StubClaw(
                    claw_id=entry.get("claw_id", "unknown"),
                    description=entry.get("description", ""),
                    static_summary=entry.get("summary", ""),
                    static_facts=entry.get("facts", []),
                )
            registry.add(instance)
        return registry

    def to_dict(self) -> dict[str, Any]:
        """Export registry as a dict (for serialization)."""
        return {
            "claws": [
                {"claw_id": c.claw_id, "description": c.description}
                for c in self._claws.values()
            ]
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Export registry as JSON. Optionally write to file."""
        data = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(data)
        return data


class _StubClaw(BaseClaw):
    """A claw created from config with static data (no module/class)."""

    def __init__(
        self,
        claw_id: str = "stub",
        description: str = "",
        static_summary: str = "",
        static_facts: list[dict[str, str]] | None = None,
    ):
        self.claw_id = claw_id
        self.description = description
        self._summary = static_summary or description
        self._facts = static_facts or []

    async def run(self, input: str) -> StateObject:
        from .models import Fact

        facts = [Fact(key=f.get("key", ""), value=f.get("value", "")) for f in self._facts]
        return StateObject(
            claw_id=self.claw_id,
            summary=self._summary,
            key_facts=facts,
            raw_context=f"Static claw: {self.description}",
            token_count=0,
        )


def get_default_registry() -> InternalRegistry:
    """Get the module-level default registry (used by @register decorator)."""
    return _get_default_registry()


def reset_default_registry() -> None:
    """Reset the default registry (useful for tests)."""
    global _default_registry
    _default_registry = InternalRegistry()
