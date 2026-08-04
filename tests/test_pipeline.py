"""Tests for the RecommenderPipeline: generation, validation retry, fallback.

Gemini is replaced with a fake so tests run offline and free.
"""

import pytest

from src.catalog import SongCatalog
from src.fallback import FallbackScorer
from src.pipeline import RecommenderPipeline
from src.retriever import Retriever
from src.validator import Validator


class FakeGenerator:
    """Stand-in for GeminiGenerator with scripted responses."""

    def __init__(self, responses=None, available=True):
        self.responses = list(responses or [])
        self.available = available
        self.calls = 0

    def is_available(self):
        return self.available

    def generate(self, query, candidates, k=5, correction=""):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(scope="module")
def songs():
    return SongCatalog().load_songs()


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


def test_recommendations_never_leave_the_catalog(songs):
    """No matter the path, recommended titles must exist in the catalog."""
    catalog_titles = {s.title for s in songs}
    pipeline = make_pipeline(songs, FakeGenerator(available=False))
    for query in ["sad slow classical", "party dance", "acoustic folk evening"]:
        for rec in pipeline.recommend(query, k=5):
            assert rec.song.title in catalog_titles
