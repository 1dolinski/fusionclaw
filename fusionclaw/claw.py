from __future__ import annotations

from abc import ABC, abstractmethod

from .models import StateObject


class BaseClaw(ABC):
    """Base class for all claws (specialist agents).

    Subclass this, set claw_id and description, implement run().
    That's the entire interface.
    """

    claw_id: str = "base"
    description: str = ""

    @abstractmethod
    async def run(self, input: str) -> StateObject:
        """Do specialist work and return structured state.

        Do NOT generate conversational output. Return a StateObject
        with your findings as structured data.
        """
        ...
