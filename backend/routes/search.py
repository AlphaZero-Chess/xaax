import json
import logging
import os
from typing import List, Optional

import httpx
from fastapi import APIRouter, Query
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

# -----------------------------
# Configuration
# -----------------------------
GOOGLE_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX = os.environ.get("GOOGLE_CSE_CX")

# LLM fallback (Emergent)
_llm_client = AsyncOpenAI(
    api_key=os.environ.get("EMERGENT_LLM_KEY"),
    base_url="https://api.emergent.sh/v1",
)


# -----------------------------
# Response models
# -----------------------------
class SuggestionsResponse(BaseModel):
    suggestions: List[str]
    query: str


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    display_link: Optional[str] = None
    thumbnail: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: Optional[str] = None
    search_time: Optional[float] = None


# -----------------------------
# Google APIs
# -----------------------------
async def _google_autocomplete(q: str, limit: int) -> List[str]:
    """Fetch suggestions from Google's free suggest endpoint (no API key).

    Uses: https://suggestqueries.google.com/complete/search?client=chrome&q=...
    """
    timeout = httpx.Timeout(3.0, connect=2.0)
    url = "https://suggestqueries.google.com/complete/search"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
    }

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        r = await client.get(url, params={"client": "chrome", "q": q})
        r.raise_for_status()

        data = r.json()
        if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list):
            items = [s for s in data[1] if isinstance(s, str)]
            return items[:limit]

    return []


async def _google_cse_search(q: str, num: int = 10, start: int = 1) -> dict:
    """Perform actual Google search using Custom Search Engine API."""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        raise ValueError("Google CSE API credentials not configured")

    timeout = httpx.Timeout(10.0, connect=5.0)
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": q,
        "num": min(num, 10),  # Max 10 per request
        "start": start,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def _llm_suggestions(q: str, limit: int) -> List[str]:
    """Fallback to LLM for suggestions."""
    response = await _llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search suggestion assistant. Given a partial search query, "
                    "provide relevant autocomplete suggestions. Return ONLY a JSON array of strings."
                ),
            },
            {"role": "user", "content": f"Provide {limit} search suggestions for: \"{q}\""},
        ],
        max_tokens=200,
        temperature=0.7,
    )

    content = (response.choices[0].message.content or "").strip()

    try:
        if content.startswith("["):
            arr = json.loads(content)
        else:
            start = content.find("[")
            end = content.rfind("]") + 1
            arr = json.loads(content[start:end]) if start >= 0 and end > start else []

        if isinstance(arr, list):
            return [s for s in arr if isinstance(s, str)][:limit]
    except Exception:
        pass

    return [
        f"{q} tutorial",
        f"{q} example",
        f"{q} documentation",
        f"how to {q}",
        f"{q} guide",
    ][:limit]


# -----------------------------
# Endpoints
# -----------------------------
@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_search_suggestions(q: str, limit: int = 5):
    """Get search suggestions (autocomplete).

    Priority:
      1) Google Suggest (free, no key)
      2) LLM suggestions via EMERGENT (fallback)
      3) Local heuristic fallback
    """
    if not q or len(q.strip()) < 2:
        return SuggestionsResponse(suggestions=[], query=q)

    q = q.strip()
    limit = max(1, min(int(limit), 10))

    # 1) Google suggest
    try:
        suggestions = await _google_autocomplete(q, limit)
        if suggestions:
            return SuggestionsResponse(suggestions=suggestions[:limit], query=q)
    except Exception as e:
        logger.warning(f"Google suggest failed (fallback to LLM): {e}")

    # 2) LLM fallback
    try:
        suggestions = await _llm_suggestions(q, limit)
        return SuggestionsResponse(suggestions=suggestions[:limit], query=q)
    except Exception as e:
        logger.error(f"LLM suggestions failed: {e}")

    # 3) final fallback
    fallback = [
        f"{q} tutorial",
        f"{q} example",
        f"{q} documentation",
        f"how to {q}",
        f"{q} guide",
    ]
    return SuggestionsResponse(suggestions=fallback[:limit], query=q)


@router.get("/query", response_model=SearchResponse)
async def search_query(
    q: str,
    num: int = Query(default=10, ge=1, le=10),
    start: int = Query(default=1, ge=1)
):
    """Perform actual Google search using Custom Search Engine API.
    
    This returns real search results from Google, not just suggestions.
    """
    if not q or len(q.strip()) < 2:
        return SearchResponse(results=[], query=q)

    q = q.strip()

    # Check if CSE is configured
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        logger.warning("Google CSE not configured, returning empty results")
        return SearchResponse(
            results=[],
            query=q,
            total_results="0",
            search_time=0.0
        )

    try:
        data = await _google_cse_search(q, num=num, start=start)
        
        results = []
        for item in data.get("items", []):
            thumbnail = None
            pagemap = item.get("pagemap", {})
            if pagemap:
                cse_thumbnail = pagemap.get("cse_thumbnail", [])
                if cse_thumbnail and len(cse_thumbnail) > 0:
                    thumbnail = cse_thumbnail[0].get("src")
                elif pagemap.get("metatags"):
                    for tag in pagemap["metatags"]:
                        if tag.get("og:image"):
                            thumbnail = tag.get("og:image")
                            break

            results.append(SearchResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
                display_link=item.get("displayLink"),
                thumbnail=thumbnail
            ))

        search_info = data.get("searchInformation", {})
        return SearchResponse(
            results=results,
            query=q,
            total_results=search_info.get("formattedTotalResults"),
            search_time=search_info.get("searchTime")
        )

    except httpx.HTTPStatusError as e:
        logger.error(f"Google CSE HTTP error: {e.response.status_code} - {e.response.text}")
        return SearchResponse(results=[], query=q, total_results="0")
    except Exception as e:
        logger.error(f"Google CSE search failed: {e}")
        return SearchResponse(results=[], query=q, total_results="0")


@router.get("/status")
async def search_status():
    """Check if Google CSE is properly configured."""
    return {
        "cse_configured": bool(GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX),
        "llm_configured": bool(os.environ.get("EMERGENT_LLM_KEY")),
    }
