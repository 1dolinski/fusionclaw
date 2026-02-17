from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from .claw import BaseClaw
from .fuser import ContextFuser
from .models import FusedContext, SynthesisResult, TokenUsage

SYSTEM_PROMPT = """You are a synthesis engine. You receive a fused context window \
containing structured knowledge from multiple specialist agents (claws).

Your job:
1. Analyze ALL context blocks — both FULL and COMPRESSED fidelity.
2. Synthesize a coherent, specific answer to the user's query.
3. Cite facts from the context. Do not invent information.
4. If blocks are marked COMPRESSED, note that some detail may be missing.

Be direct. No filler."""


class Orchestrator:
    """Takes claws + a query, runs them in parallel, fuses contexts, synthesizes.

    This is the entire API surface:
        orch = Orchestrator(claws=[MyClaw(), OtherClaw()])
        result = await orch.query("my question")
        print(result.answer)
    """

    def __init__(
        self,
        claws: list[BaseClaw],
        model: str = "gpt-4o",
        token_budget: int = 120_000,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.claws = claws
        self.model = model
        self.fuser = ContextFuser(token_budget=token_budget, model=model)

        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    async def query(
        self,
        user_input: str,
        *,
        priorities: dict[str, int] | None = None,
    ) -> SynthesisResult:
        """Run all claws in parallel, fuse their contexts, synthesize an answer."""

        # 1. Parallel execution — all claws run simultaneously
        states = await asyncio.gather(*[claw.run(user_input) for claw in self.claws])

        # 2. Fuse contexts into a single window
        fused = self.fuser.fuse(list(states), priorities=priorities)

        # 3. Synthesize via single LLM call
        return await self._synthesize(fused, user_input)

    async def _synthesize(
        self, fused: FusedContext, user_query: str
    ) -> SynthesisResult:
        """Make a single LLM call on the fused context."""
        prompt = self.fuser.build_prompt(fused, user_query)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        choice = response.choices[0]
        usage = response.usage

        return SynthesisResult(
            answer=choice.message.content or "",
            fused_context=fused,
            model=self.model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
        )
