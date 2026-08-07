"""Optional disk cache so a live demo never depends on the API being up.

The Streamlit app checks this cache before calling Gemini. Warm it with the
queries you plan to demo:

    python -m src.warmup

Cached entries survive app restarts and browser refreshes, so a rate limit
(or no network at all) can't derail a presentation. Delete .demo_cache.json
to force live calls again. Entries store song ids, not song objects, so a
changed catalog simply misses the cache rather than showing stale songs.
"""

import json
from pathlib import Path
from typing import Optional

from src.catalog import PROJECT_ROOT

CACHE_PATH = PROJECT_ROOT / ".demo_cache.json"


def _load_all() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get(key: str) -> Optional[dict]:
    return _load_all().get(key)


def put(key: str, value: dict) -> None:
    data = _load_all()
    data[key] = value
    tmp = CACHE_PATH.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        tmp.replace(CACHE_PATH)  # atomic: never leaves a half-written cache
    except OSError:
        pass  # caching is best-effort; never break a request over it


def query_key(query: str, k: int) -> str:
    return f"q::{query.strip().lower()}::{k}"


def similar_key(song_ids: list, k: int) -> str:
    return "sim::" + ",".join(str(i) for i in sorted(song_ids)) + f"::{k}"


def beyond_key(query: str) -> str:
    return f"beyond::{query.strip().lower()}"


def encode_recommendations(candidates, recommendations, source, note) -> dict:
    """Store ids and text only, so the cache is small and catalog-checked."""
    return {
        "candidates": [
            {"id": c.song.id, "score": c.relevance_score, "matched": c.matched_terms}
            for c in candidates
        ],
        "recommendations": [
            {"id": r.song.id, "rank": r.rank, "explanation": r.explanation,
             "source": r.source}
            for r in recommendations
        ],
        "source": source,
        "note": note,
    }


def decode_recommendations(entry: dict, songs_by_id: dict):
    """Rehydrate a cached entry, or return None if any song id is unknown."""
    from src.index import Candidate
    from src.pipeline import Recommendation

    try:
        candidates = [
            Candidate(song=songs_by_id[c["id"]], relevance_score=c["score"],
                      matched_terms=c["matched"])
            for c in entry["candidates"]
        ]
        recommendations = [
            Recommendation(song=songs_by_id[r["id"]], rank=r["rank"],
                           explanation=r["explanation"], source=r["source"])
            for r in entry["recommendations"]
        ]
    except KeyError:
        return None  # catalog changed since caching; treat as a miss
    return candidates, recommendations, entry["source"], entry["note"]
