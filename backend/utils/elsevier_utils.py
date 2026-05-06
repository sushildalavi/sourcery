import logging
import os
import random
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ELSEVIER_SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"
MAX_RETRIES = 3


def _backoff(attempt: int) -> float:
    return min(2 ** attempt + random.random(), 8.0)


def fetch_from_elsevier(query: str, limit: Optional[int] = None, year_from: Optional[int] = None, year_to: Optional[int] = None) -> List[Dict]:
    elsevier_key = os.getenv("ELSEVIER_API_KEY")
    elsevier_max = int(os.getenv("ELSEVIER_MAX_RESULTS", "30")) or 30
    request_timeout = float(os.getenv("ELSEVIER_TIMEOUT", "8"))
    if not elsevier_key:
        logger.debug("ELSEVIER_API_KEY not set; skipping Elsevier fetch.")
        return []
    remaining = limit if limit is not None else elsevier_max
    if remaining <= 0:
        return []

    search_query = query
    date_parts = []
    if year_from:
        date_parts.append(f"PUBYEAR > {int(year_from) - 1}")
    if year_to:
        date_parts.append(f"PUBYEAR < {int(year_to) + 1}")
    if date_parts:
        search_query = f"{query} AND {' AND '.join(date_parts)}"

    params: Dict[str, str] = {
        "query": search_query,
        "count": str(remaining),
        "sort": "relevancy",
        "httpAccept": "application/json",
    }
    headers = {
        "X-ELS-APIKey": elsevier_key,
        "Accept": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ELSEVIER_SCOPUS_URL, params=params, headers=headers, timeout=request_timeout)
            if resp.status_code in {401, 403}:
                logger.warning("Elsevier auth failed (%s); will retry next request.", resp.status_code)
                return []
            resp.raise_for_status()
            data = resp.json()
            entries = data.get("search-results", {}).get("entry", []) or []
            results = []
            for e in entries:
                if e.get("@_fa") == "false" or "error" in e:
                    continue
                yr = e.get("prism:coverDate", "")[:4]
                try:
                    yr_int = int(yr) if yr else None
                except Exception:
                    yr_int = None

                authors_raw = e.get("author", []) or []
                if not isinstance(authors_raw, list):
                    authors_raw = []
                authors = []
                for a in authors_raw:
                    if isinstance(a, dict):
                        name = a.get("authname") or a.get("given-name", "") + " " + a.get("surname", "")
                        authors.append({"display_name": name.strip()})

                doi = e.get("prism:doi")
                url = None
                links = e.get("link", [])
                if isinstance(links, list):
                    for link in links:
                        if isinstance(link, dict) and link.get("@ref") == "scopus":
                            url = link.get("@href")
                            break
                    if not url and links:
                        url = links[0].get("@href") if isinstance(links[0], dict) else None
                if not url and doi:
                    url = f"https://doi.org/{doi}"

                results.append(
                    {
                        "id": e.get("dc:identifier") or e.get("eid"),
                        "title": e.get("dc:title"),
                        "year": yr_int,
                        "doi": doi,
                        "abstract": e.get("dc:description"),
                        "authors": authors,
                        "url": url,
                        "concepts": [],
                        "source": "elsevier",
                    }
                )
            return results
        except requests.RequestException as exc:
            logger.warning("Elsevier request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(_backoff(attempt))
    return []
