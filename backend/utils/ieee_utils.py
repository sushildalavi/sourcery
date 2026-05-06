import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

IEEE_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
MAX_RETRIES = int(os.getenv("IEEE_MAX_RETRIES", "1")) or 1

# Simple in-process circuit breaker: once IEEE returns 401/403 (expired key),
# skip all subsequent calls for CIRCUIT_BREAKER_SECONDS to avoid wasting the
# per-variant 100-400ms each request takes.
_CIRCUIT_BREAKER_SECONDS = int(os.getenv("IEEE_CIRCUIT_BREAKER_SECONDS", "600")) or 600
_auth_fail_until: float = 0.0


def _backoff(attempt: int) -> float:
    return min(1.0 + random.random(), 2.0)


def fetch_from_ieee(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    global _auth_fail_until
    ieee_key = os.getenv("IEEE_API_KEY")
    ieee_max = int(os.getenv("IEEE_MAX_RESULTS", "30")) or 30
    request_timeout = float(os.getenv("IEEE_TIMEOUT", "10"))
    if not ieee_key:
        logger.debug("IEEE_API_KEY not set; skipping IEEE fetch.")
        return []
    if time.time() < _auth_fail_until:
        return []
    remaining = limit if limit is not None else ieee_max
    if remaining <= 0:
        return []
    # Prefer relevance order — publication_year sort buries seminal papers.
    params: Dict[str, str] = {
        "querytext": query,
        "apikey": ieee_key,
        "max_records": str(remaining),
        "sort_order": "desc",
        "sort_field": "relevance",
    }
    if year_from:
        params["start_year"] = str(int(year_from))
    if year_to:
        params["end_year"] = str(int(year_to))

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(IEEE_URL, params=params, timeout=request_timeout)
            if resp.status_code in {401, 403}:
                _auth_fail_until = time.time() + _CIRCUIT_BREAKER_SECONDS
                logger.warning(
                    "IEEE auth failed (%s); circuit-breaker active for %ss.",
                    resp.status_code, _CIRCUIT_BREAKER_SECONDS,
                )
                return []
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", []) or []
            results = []
            for a in articles:
                yr = a.get("publication_year")
                if year_from and yr and int(yr) < int(year_from):
                    continue
                if year_to and yr and int(yr) > int(year_to):
                    continue

                author_list = a.get("authors", {})
                if isinstance(author_list, dict):
                    author_list = author_list.get("authors", [])
                if not isinstance(author_list, list):
                    author_list = []
                authors = [
                    {"display_name": auth.get("full_name", "")}
                    for auth in author_list
                    if isinstance(auth, dict)
                ]

                index_terms = a.get("index_terms", {})
                concepts = []
                if isinstance(index_terms, dict):
                    for key in ("ieee_terms", "author_terms", "mesh_terms"):
                        terms_obj = index_terms.get(key, {})
                        if isinstance(terms_obj, dict):
                            concepts.extend(terms_obj.get("terms", []))
                elif isinstance(index_terms, list):
                    concepts = index_terms

                results.append(
                    {
                        "id": a.get("article_number"),
                        "title": a.get("title"),
                        "year": yr,
                        "doi": a.get("doi"),
                        "abstract": a.get("abstract"),
                        "authors": authors,
                        "url": a.get("pdf_url") or a.get("html_url"),
                        "concepts": concepts,
                        "source": "ieee",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("IEEE request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []
