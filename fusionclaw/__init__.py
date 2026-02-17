"""FusionClaw — context fusion beats agent chat."""

from .claw import BaseClaw
from .erc8004 import ERC8004Claw, parse_erc8004_metadata, register_claw, to_erc8004_metadata
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

__all__ = [
    "BaseClaw",
    "ContextBlock",
    "ContextFuser",
    "ERC8004Claw",
    "Fact",
    "FusedContext",
    "Orchestrator",
    "StateObject",
    "SynthesisResult",
    "TokenUsage",
    "parse_erc8004_metadata",
    "register_claw",
    "to_erc8004_metadata",
]
