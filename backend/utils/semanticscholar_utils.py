import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# Bulk endpoint supports sort=citationCount:desc — use it to surface seminal
# papers the default /search endpoint misses entirely for well-known concepts.
S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
# Keep default small to reduce 429s if no API key.
# citationCount + influentialCitationCount feed the seminal-paper boost in
# public_search._rerank; without them, highly-cited papers like
# "Attention Is All You Need" rank below recent preprints on the same topic.
S2_FIELDS = "title,year,abstract,externalIds,authors,venue,fieldsOfStudy,url,citationCount,influentialCitationCount"
MAX_RETRIES = int(os.getenv("S2_MAX_RETRIES", "2")) or 2


def _backoff(attempt: int) -> float:
    return min(2**attempt + random.random(), 8.0)


def fetch_from_s2(
    query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None
) -> List[Dict]:
    s2_max = int(os.getenv("S2_MAX_RESULTS", "20")) or 20
    s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    request_timeout = float(os.getenv("S2_TIMEOUT", "10"))
    remaining = limit if limit is not None else s2_max
    if remaining <= 0:
        return []
    params: Dict[str, str] = {
        "query": query,
        "limit": str(remaining),
        "fields": S2_FIELDS,
    }
    headers = {}
    if s2_api_key:
        headers["x-api-key"] = s2_api_key
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(S2_URL, params=params, headers=headers, timeout=request_timeout)
            if resp.status_code == 429:
                # Too many requests: back off more aggressively
                sleep_for = min(2**attempt + random.random(), 12.0)
                logger.warning(
                    "Semantic Scholar 429 (attempt %s/%s); backing off %.1fs", attempt, MAX_RETRIES, sleep_for
                )
                time.sleep(sleep_for)
                continue
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", [])
            results = []
            for it in items:
                y = it.get("year")
                if year_from and y and y < int(year_from):
                    continue
                if year_to and y and y > int(year_to):
                    continue
                doi = None
                ext = it.get("externalIds") or {}
                if isinstance(ext, dict):
                    doi = ext.get("DOI")
                results.append(
                    {
                        "id": it.get("paperId"),
                        "title": it.get("title"),
                        "year": y,
                        "doi": doi,
                        "abstract": it.get("abstract"),
                        "concepts": it.get("fieldsOfStudy") or [],
                        "authors": [{"display_name": a.get("name")} for a in it.get("authors", [])],
                        "url": it.get("url"),
                        "citation_count": it.get("citationCount") or 0,
                        "influential_citation_count": it.get("influentialCitationCount") or 0,
                        "source": "semanticscholar",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("Semantic Scholar request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []


def fetch_seminal_from_s2(query: str, limit: int = 5) -> List[Dict]:
    """Fetch the top-cited papers matching `query` via S2's bulk endpoint.

    The regular `paper/search` endpoint is relevance-ordered and drops seminal
    highly-cited papers in favor of recent preprints. The bulk endpoint
    supports `sort=citationCount:desc`, which reliably surfaces foundational
    work like Vaswani 2017 for "transformer", Goodfellow 2014 for "GAN", etc.

    Returns [] if the API call fails or no key is configured — this is a
    best-effort enrichment, not a hard dependency.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    timeout = float(os.getenv("S2_SEMINAL_TIMEOUT", "8"))
    if not query or not query.strip() or limit <= 0:
        return []
    params = {
        "query": query.strip(),
        "fields": S2_FIELDS,
        "sort": "citationCount:desc",
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    try:
        resp = requests.get(S2_BULK_URL, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 429:
            logger.warning("S2 seminal 429 — skipping")
            return []
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.warning("S2 seminal request failed: %s", exc)
        return []

    rows = (data.get("data") or [])[:limit]
    results: List[Dict] = []
    for it in rows:
        doi = None
        ext = it.get("externalIds") or {}
        if isinstance(ext, dict):
            doi = ext.get("DOI")
        results.append(
            {
                "id": it.get("paperId"),
                "title": it.get("title"),
                "year": it.get("year"),
                "doi": doi,
                "abstract": it.get("abstract"),
                "concepts": it.get("fieldsOfStudy") or [],
                "authors": [{"display_name": a.get("name")} for a in it.get("authors", [])],
                "url": it.get("url"),
                "citation_count": it.get("citationCount") or 0,
                "influential_citation_count": it.get("influentialCitationCount") or 0,
                "source": "semanticscholar",
            }
        )
    return results
