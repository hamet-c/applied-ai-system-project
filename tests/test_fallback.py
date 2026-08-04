"""Tests for the rule-based FallbackScorer (query -> prefs -> scoring)."""

import pytest

from src.catalog import SongCatalog
from src.fallback import FallbackScorer


@pytest.fixture(scope="module")
def scorer() -> FallbackScorer:
    return FallbackScorer(SongCatalog().load_songs())


def test_derive_prefs_finds_genre_mood_and_energy(scorer):
    prefs = scorer.derive_prefs("chill lofi beats for studying")
    assert prefs["genre"] == "lofi"
    assert prefs["mood"] == "chill"
    assert prefs["energy"] == 0.3


def test_derive_prefs_high_energy_and_multiword_genre(scorer):
    prefs = scorer.derive_prefs("energetic indie pop for a workout")
    assert prefs["genre"] == "indie pop"  # multi-word genre wins over "pop"
    assert prefs["energy"] == 0.9


def test_derive_prefs_mood_synonym(scorer):
    prefs = scorer.derive_prefs("something for date night")
    assert prefs.get("mood") == "romantic"


def test_derive_prefs_acoustic_preference(scorer):
    assert scorer.derive_prefs("acoustic folk")["likes_acoustic"] is True
    assert scorer.derive_prefs("electronic synth party")["likes_acoustic"] is False


def test_recommend_returns_k_sorted_picks(scorer):
    picks = scorer.recommend("happy pop", candidates=None, k=3)
    assert len(picks) == 3
    scores = [p["score"] for p in picks]
    assert scores == sorted(scores, reverse=True)
    assert picks[0]["song"].genre == "pop"


def test_recommend_without_candidates_uses_full_catalog(scorer):
    picks = scorer.recommend("melancholic classical", candidates=None, k=5)
    assert picks[0]["song"].title == "Autumn Nocturne"
