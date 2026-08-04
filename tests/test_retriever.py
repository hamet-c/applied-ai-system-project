"""Tests for the retrieval layer (SongIndex + Retriever)."""

import pytest

from src.catalog import SongCatalog
from src.retriever import Retriever


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    songs = SongCatalog().load_songs()
    return Retriever(songs, top_n=8)


def test_study_query_surfaces_calm_study_songs(retriever):
    candidates = retriever.retrieve("calm acoustic songs for studying late at night")
    assert candidates, "retrieval should find matches for a study query"
    # The catalog's dedicated study songs should dominate the top results;
    # exact ordering within that cluster is not asserted (it's fuzzy).
    study_genres = {"lofi", "ambient", "classical"}
    assert all(c.song.genre in study_genres for c in candidates[:4])
    assert "Library Rain" in [c.song.title for c in candidates]


def test_workout_query_surfaces_high_energy_songs(retriever):
    candidates = retriever.retrieve("high energy workout music")
    top_titles = [c.song.title for c in candidates[:3]]
    assert set(top_titles) & {"Neon Warehouse", "Gym Hero", "Storm Runner", "Iron Verdict"}
    assert candidates[0].song.energy > 0.65


def test_related_genre_gets_partial_credit(retriever):
    # The v1 model card flagged that a rock fan got zero credit for metal.
    # The index adds related genres, so a metal query should surface rock too.
    candidates = retriever.retrieve("metal")
    titles = [c.song.title for c in candidates]
    assert "Iron Verdict" in titles[:1]  # exact genre still wins
    assert "Storm Runner" in titles      # related genre (rock) gets credit


def test_top_n_is_respected():
    songs = SongCatalog().load_songs()
    small = Retriever(songs, top_n=3)
    assert len(small.retrieve("chill relaxing evening")) <= 3


def test_candidates_expose_matched_terms(retriever):
    candidates = retriever.retrieve("happy pop")
    assert all(c.matched_terms for c in candidates)
    assert all(c.relevance_score > 0 for c in candidates)


def test_nonsense_query_returns_empty(retriever):
    assert retriever.retrieve("xyzzy quux flurble") == []


def test_paraphrased_sleep_query_finds_low_energy_songs(retriever):
    # Regression test: this query once retrieved ZERO candidates because the
    # index knew "sleep" but not "asleep".
    candidates = retriever.retrieve("music to fall asleep to")
    assert candidates, "paraphrased sleep query should retrieve something"
    assert all(c.song.energy < 0.5 for c in candidates[:2])
