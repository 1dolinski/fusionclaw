"""FusionClaw — context fusion beats agent chat."""

from .claw import BaseClaw
from .claws import CodeAnalyzerClaw, WebSearchClaw
from .erc8004 import (
    ERC8004Claw,
    parse_erc8004_metadata,
    register_claw_onchain,
    to_erc8004_metadata,
)
from .fuser import ContextFuser
from .models import (
    ContextBlock,
    Fact,
    FusedContext,
    StateObject,
    SynthesisResult,
    TokenUsage,
)
from .orchestrator import Orchestrator
from .registry import InternalRegistry, get_default_registry, register, reset_default_registry

__all__ = [
    "BaseClaw",
    "CodeAnalyzerClaw",
    "ContextBlock",
    "ContextFuser",
    "ERC8004Claw",
    "Fact",
    "FusedContext",
    "InternalRegistry",
    "Orchestrator",
    "StateObject",
    "SynthesisResult",
    "TokenUsage",
    "WebSearchClaw",
    "get_default_registry",
    "parse_erc8004_metadata",
    "register",
    "register_claw_onchain",
    "reset_default_registry",
    "to_erc8004_metadata",
]
