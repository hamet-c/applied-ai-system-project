"""Original Module 1-3 rule-based recommender (VibeFinder 1.0).

This scoring recipe now lives on as the FallbackScorer's engine inside the
RAG pipeline (src/pipeline.py): it runs when no Gemini API key is set or
when LLM generation fails validation.

Song and UserProfile moved to src/catalog.py; re-exported here so existing
imports and tests keep working.
"""

import csv
from dataclasses import asdict
from typing import Dict, List, Tuple

from src.catalog import Song, UserProfile  # noqa: F401  (re-exported)


# Scoring weights. Adjust these to run experiments.
# Default recipe: genre 2.0, mood 1.0, energy up to 1.0, acoustic 0.5.
GENRE_WEIGHT = 2.0
MOOD_WEIGHT = 1.0
ENERGY_WEIGHT = 1.0
ACOUSTIC_WEIGHT = 0.5


def load_songs(csv_path: str) -> List[Dict]:
    """Read the CSV into a list of song dicts, converting numeric columns."""
    int_fields = ("id", "tempo_bpm")
    float_fields = ("energy", "valence", "danceability", "acousticness")

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against the user's prefs and return (score, reasons)."""
    score = 0.0
    reasons: List[str] = []

    # Genre match: strongest signal.
    if song["genre"] == user_prefs.get("genre"):
        score += GENRE_WEIGHT
        reasons.append(f"matches your favorite genre ({song['genre']})")

    # Mood match: worth half of a genre match.
    if song["mood"] == user_prefs.get("mood"):
        score += MOOD_WEIGHT
        reasons.append(f"matches your mood ({song['mood']})")

    # Energy closeness: continuous tie-breaker, up to ENERGY_WEIGHT.
    if "energy" in user_prefs:
        closeness = (1.0 - abs(user_prefs["energy"] - song["energy"])) * ENERGY_WEIGHT
        score += closeness
        if closeness > 0.8 * ENERGY_WEIGHT:
            reasons.append("very close to your energy level")

    # Acoustic preference: small nudge, only if the profile states one.
    if "likes_acoustic" in user_prefs:
        song_is_acoustic = song["acousticness"] > 0.5
        if song_is_acoustic == user_prefs["likes_acoustic"]:
            score += ACOUSTIC_WEIGHT
            reasons.append("matches your acoustic preference")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song, then return the top k as (song, score, explanation)."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons) if reasons else "no strong matches"
        scored.append((song, score, explanation))

    ranked = sorted(scored, key=lambda entry: entry[1], reverse=True)
    return ranked[:k]


class Recommender:
    """OOP wrapper over the scoring recipe, working on Song dataclasses."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    @staticmethod
    def _prefs_from_profile(user: UserProfile) -> Dict:
        return {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = self._prefs_from_profile(user)
        scored = [(song, score_song(prefs, asdict(song))[0]) for song in self.songs]
        scored.sort(key=lambda entry: entry[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = self._prefs_from_profile(user)
        _, reasons = score_song(prefs, asdict(song))
        return "; ".join(reasons) if reasons else "no strong matches"
