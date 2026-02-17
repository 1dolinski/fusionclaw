#!/usr/bin/env python3
"""Minimal example: two claws fused with one query.

Run:
    pip install -e ".[dev]"
    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""

import asyncio

from fusionclaw import BaseClaw, Fact, Orchestrator, StateObject


class PricingClaw(BaseClaw):
    claw_id = "pricing"
    description = "Analyzes competitor pricing"

    async def run(self, input: str) -> StateObject:
        # In real usage, this would scrape/research/analyze.
        # Here we return pre-built state to show the pattern.
        return StateObject(
            claw_id=self.claw_id,
            summary="Competitor dropped enterprise pricing 15% to $85/mo. New free tier added.",
            key_facts=[
                Fact(key="enterprise_price", value="$85/mo (was $100)"),
                Fact(key="free_tier", value="Just launched, 5 users included"),
                Fact(key="annual_discount", value="20%"),
            ],
            raw_context=(
                "Detailed pricing page analysis:\n"
                "- Enterprise: $85/user/mo (reduced from $100 on Jan 15)\n"
                "- Pro: $45/user/mo (unchanged)\n"
                "- Starter: $15/user/mo (unchanged)\n"
                "- Free: $0 for up to 5 users (NEW)\n"
                "- Annual billing saves 20%\n"
                "- Landing page now says 'Most affordable enterprise solution'\n"
                "- Comparison chart directly targets us and CompetitorY"
            ),
            token_count=120,
        )


class ProductClaw(BaseClaw):
    claw_id = "product"
    description = "Analyzes competitor product features"

    async def run(self, input: str) -> StateObject:
        return StateObject(
            claw_id=self.claw_id,
            summary="Competitor shipped AI features and SOC 2. Mobile app redesigned. API v2 launched.",
            key_facts=[
                Fact(key="ai_features", value="Copilot, auto-categorization, NL search"),
                Fact(key="compliance", value="SOC 2 Type II certified"),
                Fact(key="api", value="REST v2 + GraphQL beta"),
                Fact(key="mobile", value="4.5 stars (was 3.2)"),
            ],
            raw_context=(
                "Product changelog analysis (last 90 days):\n"
                "- AI Copilot: in-app assistant for workflow automation\n"
                "- Natural language search across all data\n"
                "- SOC 2 Type II certification achieved\n"
                "- HIPAA compliance in progress (ETA Q2)\n"
                "- Mobile app completely redesigned, React Native\n"
                "- REST API v2 with breaking changes from v1\n"
                "- GraphQL API in public beta\n"
                "- 15 new integrations added (Salesforce, HubSpot, etc.)\n"
                "- Dark mode and accessibility improvements"
            ),
            token_count=150,
        )


async def main():
    orch = Orchestrator(
        claws=[PricingClaw(), ProductClaw()],
        model="gpt-4o-mini",  # cheap and fast for demo
    )

    result = await orch.query(
        "How does this competitor threaten our market position? What should we do?"
    )

    print("=== Answer ===")
    print(result.answer)
    print(f"\n=== Stats ===")
    print(f"Model: {result.model}")
    print(f"Tokens: {result.usage.total_tokens}")
    print(f"Compression: {result.fused_context.compression_applied}")
    print(f"Context blocks: {len(result.fused_context.blocks)}")


if __name__ == "__main__":
    asyncio.run(main())
