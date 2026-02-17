"""Fusion approach: merge claw contexts directly, single synthesis call.

Instead of claws chatting, both claws export structured state and the
fuser merges them into a single context window for one LLM call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from fusionclaw import BaseClaw, ContextFuser, Fact, Orchestrator, StateObject
from fusionclaw.fuser import count_tokens

FIXTURES = Path(__file__).parent / "fixtures"


class PricingClaw(BaseClaw):
    claw_id = "pricing_analyst"
    description = "Analyzes competitor pricing strategy"

    async def run(self, input: str) -> StateObject:
        raw = (FIXTURES / "pricing_research.txt").read_text()
        return StateObject(
            claw_id=self.claw_id,
            summary=(
                "Competitor X cut enterprise pricing 38% over 6 months (now $49/user/mo). "
                "New Teams tier at $29/user/mo with SSO. 90-day enterprise trial. "
                "Direct comparison marketing against our pricing. "
                "3-year price lock promotion before April 2026."
            ),
            key_facts=[
                Fact(key="enterprise_price", value="$49/user/mo (was $65, was $79)"),
                Fact(key="teams_price", value="$29/user/mo (new tier)"),
                Fact(key="starter_price", value="$12/user/mo (was $15)"),
                Fact(key="enterprise_reduction_6mo", value="38%"),
                Fact(key="annual_discount", value="20% (was 15%)"),
                Fact(key="trial_period", value="90 days enterprise (was 14)"),
                Fact(key="sso_tier", value="Teams (was Enterprise only)"),
                Fact(key="price_lock", value="3 years on annual before April 2026"),
                Fact(key="geographic_discount_india", value="40%"),
                Fact(key="migration_program", value="Free data migration from top 5 competitors"),
            ],
            raw_context=raw,
            token_count=count_tokens(raw),
        )


class ProductClaw(BaseClaw):
    claw_id = "product_analyst"
    description = "Analyzes competitor product features and roadmap"

    async def run(self, input: str) -> StateObject:
        raw = (FIXTURES / "product_research.txt").read_text()
        return StateObject(
            claw_id=self.claw_id,
            summary=(
                "Competitor X shipped 3 major updates in 90 days: AI features (Claude 3.5 Sonnet), "
                "API v3 + GraphQL beta, and SOC 2 Type II. Complete UI redesign in React. "
                "Product velocity doubled. HIPAA coming March 2026. "
                "Public roadmap shows AI agents and on-premise in 2026."
            ),
            key_facts=[
                Fact(key="ai_model", value="Claude 3.5 Sonnet via Anthropic API"),
                Fact(key="ai_latency", value="2.3s average"),
                Fact(key="api_version", value="v3 REST + GraphQL beta"),
                Fact(key="sdk_languages", value="Python, JavaScript, Go, Ruby"),
                Fact(key="soc2", value="Type II certified Jan 2026"),
                Fact(key="hipaa", value="Private beta, ETA March 2026"),
                Fact(key="uptime", value="99.95% over 90 days"),
                Fact(key="integrations", value="47 native (was 31, +52%)"),
                Fact(key="mobile_rating", value="4.6/5 App Store (was 3.8)"),
                Fact(key="onboarding_time", value="12 min (was 45 min)"),
            ],
            raw_context=raw,
            token_count=count_tokens(raw),
        )


@dataclass
class FusionMetrics:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time: float = 0.0
    answer: str = ""
    compression_applied: bool = False


async def run_fusion_approach(
    client: AsyncOpenAI, model: str, query: str
) -> FusionMetrics:
    """Run the fusion-based approach.

    Flow:
    1. Both claws run in parallel (load fixture data, no LLM call)
    2. Fuser merges both contexts into a single prompt
    3. One LLM call for synthesis

    1 LLM call total. Full context preserved.
    """
    metrics = FusionMetrics()
    start = time.monotonic()

    orch = Orchestrator(
        claws=[PricingClaw(), ProductClaw()],
        model=model,
        api_key=client.api_key,
        base_url=str(client.base_url) if client.base_url else None,
    )

    result = await orch.query(query)

    metrics.answer = result.answer
    metrics.total_tokens = result.usage.total_tokens
    metrics.prompt_tokens = result.usage.prompt_tokens
    metrics.completion_tokens = result.usage.completion_tokens
    metrics.llm_calls = 1
    metrics.compression_applied = result.fused_context.compression_applied
    metrics.wall_time = time.monotonic() - start

    return metrics
