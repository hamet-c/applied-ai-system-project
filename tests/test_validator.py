"""Tests for the hallucination Validator."""

from src.catalog import Song
from src.index import Candidate
from src.validator import Validator


def make_candidates():
    songs = [
        Song(1, "Sunrise City", "Neon Echo", "pop", "happy", 0.82, 118, 0.84, 0.79, 0.18),
        Song(2, "Library Rain", "Paper Lanterns", "lofi", "chill", 0.35, 72, 0.60, 0.58, 0.86),
    ]
    return [Candidate(song=s, relevance_score=1.0) for s in songs]


def test_valid_picks_pass():
    validator = Validator()
    picks = [{"title": "Sunrise City", "reason": "r"},
             {"title": "Library Rain", "reason": "r"}]
    assert validator.validate(picks, make_candidates()) is True
    assert validator.find_hallucinated_titles(picks, make_candidates()) == []


def test_hallucinated_title_is_caught():
    validator = Validator()
    picks = [{"title": "Sunrise City", "reason": "r"},
             {"title": "Totally Invented Song", "reason": "r"}]
    assert validator.validate(picks, make_candidates()) is False
    assert validator.find_hallucinated_titles(picks, make_candidates()) == [
        "Totally Invented Song"
    ]


def test_title_matching_is_case_insensitive():
    validator = Validator()
    picks = [{"title": "  sunrise city ", "reason": "r"}]
    assert validator.find_hallucinated_titles(picks, make_candidates()) == []


def test_duplicates_fail_validation_and_deduplicate():
    validator = Validator()
    picks = [{"title": "Sunrise City", "reason": "a"},
             {"title": "SUNRISE CITY", "reason": "b"}]
    assert validator.validate(picks, make_candidates()) is False
    assert len(validator.deduplicate(picks)) == 1


def test_empty_picks_fail():
    assert Validator().validate([], make_candidates()) is False
