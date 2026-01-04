"""Web search tool with DuckDuckGo + llama.cpp summarization.

Provides a voice-friendly web search capability that:
1. Searches DuckDuckGo (free, no API key)
2. Summarizes results with llama.cpp for concise voice output
3. Returns 1-3 sentence answers instead of raw search results

Usage:
    class VoiceAssistant(WebSearchTools, Agent):
        pass  # web_search tool is automatically available
"""

import asyncio
import logging
import time
from typing import Any

from openai import OpenAI
from livekit.agents import function_tool

from ..ndjson_log import enabled as ndjson_enabled
from ..ndjson_log import log_event

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Podsumuj poniższe wyniki wyszukiwania w 1–3 zdaniach, tak aby nadawały się do odczytu głosowego.
Bądź zwięzły i konwersacyjny. Nie dodawaj adresów URL, markdowna ani wypunktowań.
Skup się na bezpośredniej odpowiedzi na to, co użytkownik najbardziej chce wiedzieć.
Odpowiedź zawsze ma być w języku polskim.

Zapytanie: {query}

Wyniki:
{results}

Podsumowanie:"""


class WebSearchTools:
    """Mixin providing web search via DuckDuckGo with llama.cpp summarization.

    Requires the parent class to have:
    - self.llm: LlamaCppLLM instance (for model access)

    Configuration (override in subclass if needed):
    - _search_max_results: int = 5
    - _search_timeout: float = 10.0
    """

    _search_max_results: int = 5
    _search_timeout: float = 10.0

    @function_tool
    async def web_search(self, query: str) -> str:
        """Search the web for current events, news, prices, store hours, or any time-sensitive information not available from other tools.

        Args:
            query: What to search for on the web.
        """
        logger.info(f"web_search: {query}")

        try:
            t0 = time.perf_counter()
            log_event(
                hypothesis_id="WS",
                location="web_search.py:web_search",
                message="web_search start",
                data={"query_preview": (query or "")[:160], "timeout_s": self._search_timeout},
            )
            raw_results = await asyncio.wait_for(
                self._do_search(query),
                timeout=self._search_timeout
            )

            if not raw_results:
                log_event(
                    hypothesis_id="WS",
                    location="web_search.py:web_search",
                    message="web_search no results",
                    data={"elapsed_ms": int((time.perf_counter() - t0) * 1000)},
                )
                return "I couldn't find any results for that search."

            log_event(
                hypothesis_id="WS",
                location="web_search.py:web_search",
                message="web_search got results",
                data={
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "results_count": len(raw_results),
                },
            )
            t1 = time.perf_counter()
            summary = await self._summarize_results(query, raw_results)
            log_event(
                hypothesis_id="WS",
                location="web_search.py:web_search",
                message="web_search summarize done",
                data={
                    "elapsed_ms": int((time.perf_counter() - t1) * 1000),
                    "summary_preview": (summary or "")[:160],
                },
            )
            return summary

        except asyncio.TimeoutError:
            logger.warning(f"Web search timed out for query: {query}")
            log_event(
                hypothesis_id="WS",
                location="web_search.py:web_search",
                message="web_search timeout",
                data={"query_preview": (query or "")[:160], "timeout_s": self._search_timeout},
            )
            return "The search took too long. Please try a simpler query."
        except Exception as e:
            logger.error(f"Web search error: {e}", exc_info=True)
            log_event(
                hypothesis_id="WS",
                location="web_search.py:web_search",
                message="web_search exception",
                data={"error": str(e)[:200]},
            )
            return "I had trouble searching the web. Please try again."

    async def _do_search(self, query: str) -> list[dict[str, Any]]:
        """Execute DuckDuckGo search in thread pool (blocking API).

        Returns list of result dicts with 'title', 'body', 'href' keys.
        """
        from ddgs import DDGS

        def _search():
            with DDGS(timeout=self._search_timeout) as ddgs:
                return list(ddgs.text(
                    query,
                    max_results=self._search_max_results,
                    safesearch="moderate"
                ))

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)

    async def _summarize_results(
        self,
        query: str,
        results: list[dict[str, Any]]
    ) -> str:
        """Summarize search results with llama.cpp for voice-friendly output."""

        # Truncate to avoid exceeding context limits (~500 tokens total)
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")[:100]
            body = r.get("body", "")[:200]
            formatted.append(f"{i}. {title}: {body}")

        results_text = "\n".join(formatted)
        prompt = SUMMARIZE_PROMPT.format(query=query, results=results_text)

        # Use agent's model and base_url for summarization
        model = getattr(self.llm, "model", "gpt-oss-20b-mxfp4")
        base_url = getattr(self.llm, "base_url", "http://llama.home/v1")

        try:
            t0 = time.perf_counter()
            log_event(
                hypothesis_id="WS",
                location="web_search.py:_summarize_results",
                message="summarize start",
                data={"model": model, "results_count": len(results), "base_url": base_url},
            )
            client = OpenAI(
                base_url=base_url,
                api_key="not-needed",  # llama.cpp doesn't require auth
            )

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Low temp for factual output
                max_tokens=200,
            )
            # Capture summarizer response shape; some OpenAI-compatible servers return empty content.
            if ndjson_enabled():
                try:
                    ch0 = response.choices[0] if getattr(response, "choices", None) else None
                    msg0 = getattr(ch0, "message", None) if ch0 else None
                    raw = getattr(msg0, "content", None) if msg0 else None
                    raw = raw if isinstance(raw, str) else ""
                    finish_reason = getattr(ch0, "finish_reason", None) if ch0 else None
                    log_event(
                        hypothesis_id="WS",
                        location="web_search.py:_summarize_results",
                        message="summarize response meta",
                        data={
                            "choices_len": len(getattr(response, "choices", []) or []),
                            "finish_reason": str(finish_reason)[:40] if finish_reason is not None else None,
                            "raw_len": len(raw),
                            "raw_preview": raw[:160],
                        },
                    )
                except Exception:
                    pass
            summary = response.choices[0].message.content.strip() if response.choices else ""
            log_event(
                hypothesis_id="WS",
                location="web_search.py:_summarize_results",
                message="summarize ok",
                data={
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "summary_preview": (summary or "")[:160],
                },
            )
            if summary:
                return summary

            # Deterministic fallback: speak top result snippets instead of a generic "couldn't summarize".
            top = []
            for r in results[:2]:
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                if title and body:
                    top.append(f"{title}. {body}")
                elif title:
                    top.append(title)
                elif body:
                    top.append(body)
            return " ".join(top) if top else "I found results but couldn't summarize them."

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            log_event(
                hypothesis_id="WS",
                location="web_search.py:_summarize_results",
                message="summarize exception",
                data={"error": str(e)[:200]},
            )
            # Fallback: return first result's snippet
            if results:
                return results[0].get("body", "No description available.")
            return "I had trouble processing the search results."
