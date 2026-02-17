"""Chat baseline: two claws communicate via sequential LLM summarization.

This simulates how CrewAI, AutoGen, etc. work — agents chat with each other,
each generating a text summary that gets passed to the next agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class ChatMetrics:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    wall_time: float = 0.0
    answer: str = ""
    intermediate_messages: list[str] = field(default_factory=list)


async def run_chat_baseline(
    client: AsyncOpenAI, model: str, query: str
) -> ChatMetrics:
    """Run the chat-based multi-agent approach.

    Flow:
    1. Claw A reads its data, generates a summary via LLM
    2. Claw A's summary is sent as a "message" to Claw B
    3. Claw B reads Claw A's message + its own data, generates a combined summary
    4. Combined summary goes to a synthesizer for final answer

    This is 3 sequential LLM calls. Information degrades at each hop.
    """
    metrics = ChatMetrics()
    start = time.monotonic()

    pricing_data = (FIXTURES / "pricing_research.txt").read_text()
    product_data = (FIXTURES / "product_research.txt").read_text()

    # --- LLM Call 1: Claw A summarizes its findings ---
    resp1 = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a pricing analyst agent. Summarize your research findings concisely so another agent can understand them.",
            },
            {
                "role": "user",
                "content": f"Here is your research data:\n\n{pricing_data}\n\nSummarize the key findings relevant to this query: {query}",
            },
        ],
        temperature=0.3,
    )
    claw_a_summary = resp1.choices[0].message.content or ""
    metrics.intermediate_messages.append(claw_a_summary)
    if resp1.usage:
        metrics.prompt_tokens += resp1.usage.prompt_tokens
        metrics.completion_tokens += resp1.usage.completion_tokens
        metrics.total_tokens += resp1.usage.total_tokens
    metrics.llm_calls += 1

    # --- LLM Call 2: Claw B receives Claw A's summary + its own data ---
    resp2 = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a product analyst agent. Another agent has shared their pricing analysis with you. Combine it with your product research into a comprehensive summary.",
            },
            {
                "role": "user",
                "content": (
                    f"Message from Pricing Agent:\n\n{claw_a_summary}\n\n"
                    f"---\n\nHere is your own product research data:\n\n{product_data}\n\n"
                    f"Combine both analyses into a comprehensive summary for this query: {query}"
                ),
            },
        ],
        temperature=0.3,
    )
    combined_summary = resp2.choices[0].message.content or ""
    metrics.intermediate_messages.append(combined_summary)
    if resp2.usage:
        metrics.prompt_tokens += resp2.usage.prompt_tokens
        metrics.completion_tokens += resp2.usage.completion_tokens
        metrics.total_tokens += resp2.usage.total_tokens
    metrics.llm_calls += 1

    # --- LLM Call 3: Synthesizer produces final answer ---
    resp3 = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strategic analyst. Produce a detailed final analysis based on the combined research summary provided.",
            },
            {
                "role": "user",
                "content": f"Combined research summary:\n\n{combined_summary}\n\nAnswer this query in detail: {query}",
            },
        ],
        temperature=0.3,
    )
    metrics.answer = resp3.choices[0].message.content or ""
    if resp3.usage:
        metrics.prompt_tokens += resp3.usage.prompt_tokens
        metrics.completion_tokens += resp3.usage.completion_tokens
        metrics.total_tokens += resp3.usage.total_tokens
    metrics.llm_calls += 1

    metrics.wall_time = time.monotonic() - start
    return metrics
