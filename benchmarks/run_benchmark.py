#!/usr/bin/env python3
"""Benchmark harness: runs chat baseline vs fusion approach, prints metrics table.

Usage:
    python -m benchmarks.run_benchmark
    python -m benchmarks.run_benchmark --model gpt-4o-mini --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from benchmarks.chat_baseline import ChatMetrics, run_chat_baseline
from benchmarks.fusion_approach import FusionMetrics, run_fusion_approach

# Known facts embedded in fixtures that we check for in the final answer
KNOWN_FACTS = [
    "49",  # enterprise price $49
    "29",  # teams price $29
    "38%",  # enterprise reduction
    "90-day",  # trial period
    "SSO",  # SSO moved to Teams tier
    "Claude",  # AI model used
    "SOC 2",  # compliance
    "GraphQL",  # new API
    "99.95%",  # uptime
    "12 min",  # onboarding time
]

# Pricing per 1M tokens (input, output) — approximate
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


def count_facts_retained(answer: str) -> int:
    """Count how many known facts appear in the answer."""
    answer_lower = answer.lower()
    return sum(1 for fact in KNOWN_FACTS if fact.lower() in answer_lower)


def estimate_cost(
    prompt_tokens: int, completion_tokens: int, model: str
) -> float:
    """Estimate cost in USD based on token usage."""
    input_price, output_price = MODEL_PRICING.get(model, (2.50, 10.00))
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


async def run_quality_judge(
    client: AsyncOpenAI, model: str, query: str, answer: str
) -> float:
    """Use LLM-as-judge to score answer quality 1-10."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. Score the following answer on a scale of 1-10 "
                    "based on: completeness, accuracy, specificity, and actionability. "
                    "Respond with ONLY a JSON object: {\"score\": <number>, \"reason\": \"<brief reason>\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nAnswer:\n{answer}",
            },
        ],
        temperature=0.1,
    )
    text = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(text)
        return float(parsed["score"])
    except (json.JSONDecodeError, KeyError, ValueError):
        # Try to extract a number
        for word in text.split():
            try:
                val = float(word.strip(".,"))
                if 1 <= val <= 10:
                    return val
            except ValueError:
                continue
        return 5.0


def fmt_delta(chat_val: float, fusion_val: float, lower_is_better: bool = True) -> str:
    """Format a delta as a percentage with direction."""
    if chat_val == 0:
        return "N/A"
    pct = ((fusion_val - chat_val) / chat_val) * 100
    if lower_is_better:
        sign = "" if pct > 0 else ""
    else:
        sign = "+" if pct > 0 else ""
    return f"{sign}{pct:+.0f}%"


async def run_single(client: AsyncOpenAI, model: str, query: str, judge: bool = True):
    """Run one round of chat vs fusion and return metrics."""
    print("  Running chat baseline...", flush=True)
    chat = await run_chat_baseline(client, model, query)

    print("  Running fusion approach...", flush=True)
    fusion = await run_fusion_approach(client, model, query)

    chat_quality = 0.0
    fusion_quality = 0.0
    if judge:
        print("  Judging answer quality...", flush=True)
        chat_quality, fusion_quality = await asyncio.gather(
            run_quality_judge(client, model, query, chat.answer),
            run_quality_judge(client, model, query, fusion.answer),
        )

    return chat, fusion, chat_quality, fusion_quality


async def main():
    parser = argparse.ArgumentParser(description="FusionClaw Benchmark")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to average")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM quality judging")
    parser.add_argument(
        "--query",
        default="How does Competitor X's strategy threaten our market position? What are their strengths and weaknesses?",
        help="Query to benchmark",
    )
    args = parser.parse_args()

    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    print(f"\nFusionClaw Benchmark")
    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}")
    print(f"Query: {args.query}\n")

    # Accumulate metrics across runs
    chat_tokens, fusion_tokens = [], []
    chat_times, fusion_times = [], []
    chat_calls, fusion_calls = [], []
    chat_costs, fusion_costs = [], []
    chat_facts, fusion_facts = [], []
    chat_qualities, fusion_qualities = [], []

    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs}:")
        chat, fusion, cq, fq = await run_single(
            client, args.model, args.query, judge=not args.no_judge
        )

        chat_tokens.append(chat.total_tokens)
        fusion_tokens.append(fusion.total_tokens)
        chat_times.append(chat.wall_time)
        fusion_times.append(fusion.wall_time)
        chat_calls.append(chat.llm_calls)
        fusion_calls.append(fusion.llm_calls)
        chat_costs.append(estimate_cost(chat.prompt_tokens, chat.completion_tokens, args.model))
        fusion_costs.append(estimate_cost(fusion.prompt_tokens, fusion.completion_tokens, args.model))
        chat_facts.append(count_facts_retained(chat.answer))
        fusion_facts.append(count_facts_retained(fusion.answer))
        chat_qualities.append(cq)
        fusion_qualities.append(fq)

    # Average
    n = args.runs
    avg = lambda xs: sum(xs) / n

    ct, ft = avg(chat_tokens), avg(fusion_tokens)
    cw, fw = avg(chat_times), avg(fusion_times)
    cc, fc = avg(chat_calls), avg(fusion_calls)
    cco, fco = avg(chat_costs), avg(fusion_costs)
    cf, ff = avg(chat_facts), avg(fusion_facts)
    cqu, fqu = avg(chat_qualities), avg(fusion_qualities)

    total_facts = len(KNOWN_FACTS)

    print(f"\n{'='*65}")
    print(f"  RESULTS {'(averaged over ' + str(n) + ' runs)' if n > 1 else ''}")
    print(f"{'='*65}")
    print(f"")
    print(f"| {'Metric':<20} | {'Chat':>10} | {'Fusion':>10} | {'Delta':>10} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*12}|{'-'*12}|")
    print(f"| {'Total tokens':<20} | {ct:>10,.0f} | {ft:>10,.0f} | {fmt_delta(ct, ft):>10} |")
    print(f"| {'Wall time':<20} | {cw:>9.1f}s | {fw:>9.1f}s | {fmt_delta(cw, fw):>10} |")
    print(f"| {'LLM calls':<20} | {cc:>10.0f} | {fc:>10.0f} | {fmt_delta(cc, fc):>10} |")
    print(f"| {'Est. cost':<20} | ${cco:>9.4f} | ${fco:>9.4f} | {fmt_delta(cco, fco):>10} |")
    print(f"| {'Facts retained':<20} | {cf:>5.0f}/{total_facts:>4} | {ff:>5.0f}/{total_facts:>4} | {fmt_delta(cf, ff, lower_is_better=False):>10} |")
    if not args.no_judge:
        print(f"| {'Quality (1-10)':<20} | {cqu:>10.1f} | {fqu:>10.1f} | {fmt_delta(cqu, fqu, lower_is_better=False):>10} |")
    print()

    # Print answers for inspection
    if n == 1:
        print(f"\n{'='*65}")
        print("  CHAT ANSWER (last run)")
        print(f"{'='*65}")
        print(chat.answer[:2000])
        if len(chat.answer) > 2000:
            print(f"\n... ({len(chat.answer)} chars total)")

        print(f"\n{'='*65}")
        print("  FUSION ANSWER (last run)")
        print(f"{'='*65}")
        print(fusion.answer[:2000])
        if len(fusion.answer) > 2000:
            print(f"\n... ({len(fusion.answer)} chars total)")


if __name__ == "__main__":
    asyncio.run(main())
