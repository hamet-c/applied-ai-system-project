"""Rule-based fallback scorer — the original Module 1-3 recommender, adapted.

Used when there is no Gemini API key, or when generation fails validation
after a retry. It derives a rough taste profile from the free-text query
(keyword matching against the catalog's genres/moods plus energy words),
then applies the original weighted scoring recipe over the retrieved
candidates. Explanations are templated rather than generated.
"""

from dataclasses import asdict
from typing import Dict, List, Optional

from src.catalog import Song
from src.index import Candidate, tokenize
from src.recommender import score_song

# Query words that imply an energy target.
LOW_ENERGY_WORDS = {
    "calm", "chill", "chilled", "relax", "relaxed", "relaxing", "sleep",
    "quiet", "soft", "mellow", "slow", "gentle", "study", "studying",
    "focu", "focus", "focused", "peaceful", "soothing", "unwind",
}
HIGH_ENERGY_WORDS = {
    "workout", "gym", "hype", "intense", "energetic", "upbeat", "party",
    "loud", "fast", "pump", "pumped", "running", "dance", "dancing",
    "energy", "power", "powerful",
}

# Query words that map onto catalog moods.
MOOD_SYNONYMS = {
    "study": "focused", "studying": "focused", "focus": "focused",
    "relax": "relaxed", "relaxing": "relaxed", "chilled": "chill",
    "love": "romantic", "date": "romantic", "melancholy": "melancholic",
    "dream": "dreamy", "nostalgia": "nostalgic",
}


class FallbackScorer:
    """Scores retrieved candidates with the original weighted recipe."""

    def __init__(self, songs: List[Song]):
        self.songs = list(songs)
        self.known_genres = {s.genre for s in self.songs}
        self.known_moods = {s.mood for s in self.songs}

    def derive_prefs(self, query: str) -> Dict:
        """Best-effort taste profile extracted from the query text."""
        text = query.lower()
        tokens = set(tokenize(query))
        prefs: Dict = {}

        # Genre: check multi-word genres as substrings ("indie pop", "r&b").
        for genre in sorted(self.known_genres, key=len, reverse=True):
            if genre in text:
                prefs["genre"] = genre
                break

        # Mood: direct token match first, then synonyms.
        for mood in self.known_moods:
            if mood in tokens or mood in text.split():
                prefs["mood"] = mood
                break
        if "mood" not in prefs:
            for word, mood in MOOD_SYNONYMS.items():
                if word in tokens and mood in self.known_moods:
                    prefs["mood"] = mood
                    break

        if tokens & LOW_ENERGY_WORDS:
            prefs["energy"] = 0.3
        elif tokens & HIGH_ENERGY_WORDS:
            prefs["energy"] = 0.9

        if "acoustic" in tokens or "unplugged" in tokens:
            prefs["likes_acoustic"] = True
        elif tokens & {"electronic", "synth", "electric", "edm"}:
            prefs["likes_acoustic"] = False

        return prefs

    def recommend(self, query: str, candidates: Optional[List[Candidate]],
                  k: int = 5) -> List[dict]:
        """Return templated picks: [{"title", "reason", "song", "score"}].

        Scores `candidates` when retrieval found matches; otherwise falls
        back to scoring the whole catalog with the derived profile.
        """
        prefs = self.derive_prefs(query)

        if candidates:
            pool = [(c.song, c.relevance_score, c.matched_terms) for c in candidates]
        else:
            pool = [(s, 0.0, []) for s in self.songs]

        scored = []
        for song, relevance, matched in pool:
            rule_score, reasons = score_song(prefs, asdict(song))
            total = rule_score + relevance  # retrieval relevance breaks ties
            if matched:
                reasons.append("matched your search terms: " + ", ".join(matched))
            reason_text = "; ".join(reasons) if reasons else "closest available match"
            scored.append({"title": song.title, "reason": reason_text,
                           "song": song, "score": total})

        scored.sort(key=lambda e: e["score"], reverse=True)
        return scored[:k]
