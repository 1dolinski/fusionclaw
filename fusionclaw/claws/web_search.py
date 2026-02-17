"""Web search claw — searches the web via DuckDuckGo (no API key needed).

Uses the DuckDuckGo HTML search to avoid any API key requirements.
"""

from __future__ import annotations

import httpx

from ..claw import BaseClaw
from ..fuser import count_tokens
from ..models import Fact, StateObject


class WebSearchClaw(BaseClaw):
    """Searches the web for current information on a topic.

    Uses DuckDuckGo Lite (HTML) for zero-config search. No API key needed.
    """

    claw_id = "web_search"
    description = "Searches the web for current information using DuckDuckGo"

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    async def run(self, input: str) -> StateObject:
        results = await self._search(input)

        if not results:
            return StateObject(
                claw_id=self.claw_id,
                summary=f"No web results found for: {input}",
                key_facts=[],
                raw_context="Search returned no results.",
                token_count=0,
            )

        facts = []
        raw_lines = []
        for i, r in enumerate(results[: self.max_results]):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("url", "")
            facts.append(Fact(key=f"result_{i+1}", value=f"{title}: {snippet[:100]}"))
            raw_lines.append(f"[{i+1}] {title}\n    URL: {url}\n    {snippet}\n")

        raw_context = "\n".join(raw_lines)

        return StateObject(
            claw_id=self.claw_id,
            summary=f"Found {len(results)} web results for '{input}'",
            key_facts=facts,
            raw_context=raw_context,
            token_count=count_tokens(raw_context),
        )

    async def _search(self, query: str) -> list[dict[str, str]]:
        """Search DuckDuckGo Lite and parse results."""
        results: list[dict[str, str]] = []
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                resp = await client.get(
                    "https://lite.duckduckgo.com/lite/",
                    params={"q": query},
                    headers={"User-Agent": "FusionClaw/0.1 (context-fusion agent)"},
                    timeout=10.0,
                )
                resp.raise_for_status()
                html = resp.text

                # Simple parsing of DuckDuckGo Lite results
                entries = html.split('<a rel="nofollow"')
                for entry in entries[1:]:  # skip first split part
                    url = ""
                    title = ""
                    snippet = ""

                    # Extract URL
                    href_start = entry.find('href="')
                    if href_start != -1:
                        href_start += 6
                        href_end = entry.find('"', href_start)
                        url = entry[href_start:href_end]

                    # Extract title (text inside the <a> tag)
                    close_tag = entry.find(">")
                    end_a = entry.find("</a>")
                    if close_tag != -1 and end_a != -1:
                        title = entry[close_tag + 1 : end_a].strip()
                        title = _strip_html(title)

                    # Extract snippet (text after the link)
                    td_start = entry.find('<td class="result-snippet">')
                    if td_start != -1:
                        td_start += 27
                        td_end = entry.find("</td>", td_start)
                        if td_end != -1:
                            snippet = _strip_html(entry[td_start:td_end]).strip()

                    if url and title and not url.startswith("/"):
                        results.append(
                            {"url": url, "title": title, "snippet": snippet}
                        )

        except Exception:
            pass

        return results


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    import re

    return re.sub(r"<[^>]+>", "", text).strip()
