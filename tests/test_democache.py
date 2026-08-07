"""Tests for the optional demo cache (disk-backed, used by the Streamlit app)."""

from pathlib import Path

import pytest

from src import democache
from src.catalog import SongCatalog
from src.index import Candidate
from src.pipeline import Recommendation

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "songs_small.csv"


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    """Point the cache at a temp file so tests never touch the real one."""
    monkeypatch.setattr(democache, "CACHE_PATH", tmp_path / "cache.json")


@pytest.fixture(scope="module")
def songs():
    return SongCatalog(FIXTURE_CSV).load_songs()


def test_missing_key_returns_none():
    assert democache.get("q::nothing::5") is None


def test_round_trip_preserves_recommendations(songs):
    candidates = [Candidate(song=songs[0], relevance_score=0.9, matched_terms=["pop"])]
    recs = [Recommendation(song=songs[0], rank=1, explanation="great fit",
                           source="gemini")]
    key = democache.query_key("Happy Pop ", 5)

    democache.put(key, democache.encode_recommendations(
        candidates, recs, "gemini", "a note"))

    entry = democache.get(democache.query_key("happy pop", 5))  # normalized
    assert entry is not None
    got_cands, got_recs, source, note = democache.decode_recommendations(
        entry, {s.id: s for s in songs})
    assert source == "gemini" and note == "a note"
    assert got_recs[0].song.title == songs[0].title
    assert got_recs[0].explanation == "great fit"
    assert got_cands[0].matched_terms == ["pop"]


def test_unknown_song_id_is_a_cache_miss(songs):
    """A changed catalog must never resurrect stale songs."""
    recs = [Recommendation(song=songs[0], rank=1, explanation="x", source="gemini")]
    key = democache.query_key("q", 5)
    democache.put(key, democache.encode_recommendations([], recs, "gemini", ""))

    assert democache.decode_recommendations(democache.get(key), {}) is None


def test_keys_are_order_insensitive_for_playlists():
    assert democache.similar_key([3, 1, 2], 5) == democache.similar_key([1, 2, 3], 5)
    assert democache.similar_key([1, 2], 5) != democache.similar_key([1, 2], 3)


def test_corrupt_cache_file_is_ignored():
    democache.CACHE_PATH.write_text("{not json", encoding="utf-8")
    assert democache.get("anything") is None
    democache.put("k", {"candidates": [], "recommendations": [],
                        "source": "gemini", "note": ""})
    assert democache.get("k") is not None  # recovers by rewriting
