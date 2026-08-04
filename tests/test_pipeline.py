"""Tests for the RecommenderPipeline: generation, validation retry, fallback.

Gemini is replaced with a fake so tests run offline and free.
"""

from pathlib import Path

import pytest

from src.catalog import SongCatalog
from src.fallback import FallbackScorer
from src.pipeline import RecommenderPipeline
from src.retriever import Retriever
from src.validator import Validator

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "songs_small.csv"


class FakeGenerator:
    """Stand-in for GeminiGenerator with scripted responses."""

    def __init__(self, responses=None, available=True, interpretation=""):
        self.responses = list(responses or [])
        self.available = available
        self.interpretation = interpretation
        self.calls = 0
        self.interpret_calls = 0

    def is_available(self):
        return self.available

    def generate(self, query, candidates, k=5, correction=""):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def interpret_query(self, query, genres, moods):
        self.interpret_calls += 1
        return self.interpretation

    def suggest_beyond_catalog(self, query, k=3):
        return [{"title": "Real Song", "artist": "Real Artist", "reason": "fits"}][:k]


@pytest.fixture(scope="module")
def songs():
    return SongCatalog(FIXTURE_CSV).load_songs()


def make_pipeline(songs, generator):
    return RecommenderPipeline(
        retriever=Retriever(songs, top_n=8),
        generator=generator,
        validator=Validator(),
        fallback=FallbackScorer(songs),
    )


def candidate_titles(pipeline, query):
    return [c.song.title for c in pipeline.retriever.retrieve(query)]


def test_valid_generation_is_used(songs):
    query = "chill lofi for studying"
    generator = FakeGenerator()
    pipeline = make_pipeline(songs, generator)
    titles = candidate_titles(pipeline, query)
    generator.responses = [[
        {"title": titles[0], "reason": "perfect study vibe"},
        {"title": titles[1], "reason": "calm and mellow"},
    ]]

    recs = pipeline.recommend(query, k=2)

    assert pipeline.last_source == "gemini"
    assert [r.song.title for r in recs] == titles[:2]
    assert recs[0].rank == 1 and recs[0].source == "gemini"
    assert recs[0].explanation == "perfect study vibe"


def test_hallucination_triggers_retry_then_succeeds(songs):
    query = "happy pop"
    generator = FakeGenerator()
    pipeline = make_pipeline(songs, generator)
    titles = candidate_titles(pipeline, query)
    generator.responses = [
        [{"title": "Invented Banger", "reason": "x"}],          # attempt 1: bad
        [{"title": titles[0], "reason": "upbeat and bright"}],  # attempt 2: good
    ]

    recs = pipeline.recommend(query, k=1)

    assert generator.calls == 2
    assert pipeline.last_source == "gemini"
    assert recs[0].song.title == titles[0]


def test_persistent_hallucination_falls_back(songs):
    query = "happy pop"
    generator = FakeGenerator(responses=[
        [{"title": "Fake Song A", "reason": "x"}],
        [{"title": "Fake Song B", "reason": "x"}],
    ])
    pipeline = make_pipeline(songs, generator)

    recs = pipeline.recommend(query, k=3)

    assert generator.calls == 2
    assert pipeline.last_source == "fallback"
    assert all(r.source == "fallback" for r in recs)
    # Fallback picks still come from the retrieved candidate set.
    retrieved = set(candidate_titles(pipeline, query))
    assert all(r.song.title in retrieved for r in recs)


def test_no_api_key_uses_fallback(songs):
    pipeline = make_pipeline(songs, FakeGenerator(available=False))

    recs = pipeline.recommend("high energy workout music", k=3)

    assert pipeline.last_source == "fallback"
    assert "GEMINI_API_KEY" in pipeline.last_note
    assert len(recs) == 3


def test_generator_exception_falls_back(songs):
    generator = FakeGenerator(responses=[RuntimeError("api down")])
    pipeline = make_pipeline(songs, generator)

    recs = pipeline.recommend("chill lofi", k=2)

    assert pipeline.last_source == "fallback"
    assert recs, "fallback should still produce recommendations"


def test_empty_query_is_handled_gracefully(songs):
    pipeline = make_pipeline(songs, FakeGenerator(available=False))

    recs = pipeline.recommend("", k=3)

    assert len(recs) == 3  # generic picks from the whole catalog, no crash
    assert pipeline.last_source == "fallback"
    assert "No strong keyword matches" in pipeline.last_note


def test_out_of_catalog_query_is_interpreted(songs):
    """A real-world artist query retrieves nothing directly; the pipeline asks
    the generator to translate it into catalog vocabulary and retries."""
    generator = FakeGenerator(interpretation="hip-hop confident energetic")
    pipeline = make_pipeline(songs, generator)
    # Scripted pick must come from the re-retrieved candidate set.
    interpreted = [c.song.title for c in pipeline.retriever.retrieve("hip-hop confident energetic")]
    generator.responses = [[{"title": interpreted[0], "reason": "closest vibe"}]]

    recs = pipeline.recommend("songs like Drake", k=1)

    assert generator.interpret_calls == 1
    assert pipeline.last_interpretation == "hip-hop confident energetic"
    assert pipeline.last_candidates, "interpreted retrieval should find candidates"
    assert recs[0].song.title == interpreted[0]
    assert "interpreted your request" in pipeline.last_note


def test_interpretation_not_used_when_direct_matches_exist(songs):
    generator = FakeGenerator(interpretation="should not be called")
    pipeline = make_pipeline(songs, generator)
    titles = candidate_titles(pipeline, "happy pop")
    generator.responses = [[{"title": titles[0], "reason": "r"}]]

    pipeline.recommend("happy pop", k=1)

    assert generator.interpret_calls == 0
    assert pipeline.last_interpretation == ""


def test_beyond_catalog_is_separate_and_labeled_data(songs):
    pipeline = make_pipeline(songs, FakeGenerator())
    suggestions = pipeline.beyond_catalog("workout music", k=1)
    assert suggestions == [{"title": "Real Song", "artist": "Real Artist", "reason": "fits"}]
    # And it must NOT be available without the LLM.
    offline = make_pipeline(songs, FakeGenerator(available=False))
    assert offline.beyond_catalog("workout music") == []


def test_similar_excludes_playlist_and_matches_vibe(songs):
    pipeline = make_pipeline(songs, FakeGenerator(available=False))
    playlist = [s for s in songs if s.title in ("Midnight Coding", "Focus Flow")]

    recs = pipeline.recommend_similar(playlist, k=3)

    titles = [r.song.title for r in recs]
    assert len(recs) == 3
    assert not set(titles) & {"Midnight Coding", "Focus Flow"}
    # A lofi playlist should pull in the other calm/study songs first.
    assert recs[0].song.genre in {"lofi", "ambient", "classical", "jazz"}
    assert pipeline.last_source == "fallback"


def test_similar_uses_gemini_when_available(songs):
    generator = FakeGenerator()
    pipeline = make_pipeline(songs, generator)
    playlist = [s for s in songs if s.genre == "lofi"]
    seed_candidates = pipeline.retriever.index.similar(playlist, top_n=8)
    generator.responses = [[
        {"title": seed_candidates[0].song.title, "reason": "fits the playlist"},
    ]]

    recs = pipeline.recommend_similar(playlist, k=1)

    assert pipeline.last_source == "gemini"
    assert recs[0].song.title == seed_candidates[0].song.title
    assert recs[0].explanation == "fits the playlist"


def test_similar_with_empty_playlist_returns_nothing(songs):
    pipeline = make_pipeline(songs, FakeGenerator(available=False))
    assert pipeline.recommend_similar([], k=3) == []


def test_recommendations_never_leave_the_catalog(songs):
    """No matter the path, recommended titles must exist in the catalog."""
    catalog_titles = {s.title for s in songs}
    pipeline = make_pipeline(songs, FakeGenerator(available=False))
    for query in ["sad slow classical", "party dance", "acoustic folk evening"]:
        for rec in pipeline.recommend(query, k=5):
            assert rec.song.title in catalog_titles
