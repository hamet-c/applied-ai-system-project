"""RecommenderPipeline: orchestrates retrieve -> generate -> validate -> fallback.

This is the main entry point the CLI, Streamlit UI, tests and evaluation
harness all share. The flow (mirrors diagrams/architecture.mmd):

    query -> Retriever -> Candidates
          -> GeminiGenerator (grounded in candidates only)
          -> Validator (hallucinated title? retry once with a correction)
          -> on failure or no API key: FallbackScorer
          -> Recommendation[] back to the caller
"""

from dataclasses import dataclass
from typing import List, Optional

from src.catalog import Song, SongCatalog
from src.fallback import FallbackScorer
from src.generator import GeminiGenerator
from src.index import Candidate
from src.retriever import Retriever
from src.validator import Validator


@dataclass
class Recommendation:
    song: Song
    rank: int
    explanation: str
    source: str  # "gemini" or "fallback"


class RecommenderPipeline:
    """Wires the RAG components together; components are injectable for tests."""

    def __init__(self, retriever: Retriever, generator: GeminiGenerator,
                 validator: Validator, fallback: FallbackScorer):
        self.retriever = retriever
        self.generator = generator
        self.validator = validator
        self.fallback = fallback
        self.last_source: str = ""
        self.last_note: str = ""
        self.last_interpretation: str = ""
        self.last_candidates: List[Candidate] = []
        # True when the result was shaped by a transient failure (API error,
        # rate limit): callers should NOT cache it, so the next try recovers.
        self.last_transient: bool = False

    @classmethod
    def from_catalog(cls, csv_path: str = None, top_n: int = 8) -> "RecommenderPipeline":
        songs = SongCatalog(csv_path).load_songs()
        return cls(
            retriever=Retriever(songs, top_n=top_n),
            generator=GeminiGenerator(),
            validator=Validator(),
            fallback=FallbackScorer(songs),
        )

    def recommend(self, query: str, k: int = 5,
                  candidates: Optional[List[Candidate]] = None) -> List[Recommendation]:
        """Run the full RAG flow. Pass precomputed candidates to avoid re-retrieval."""
        self.last_interpretation = ""
        self.last_transient = False
        self.last_note = ""
        if candidates is None:
            candidates = self.retriever.retrieve(query)

        # No direct matches (e.g. a real-world artist): ask Gemini to
        # translate the request into catalog vocabulary and retrieve again.
        # Grounding is preserved — picks still come only from the catalog.
        if not candidates and self.generator.is_available():
            phrase = ""
            try:
                phrase = self.generator.interpret_query(
                    query,
                    genres=self._genre_vocabulary(),
                    moods=sorted({s.mood for s in self.retriever.index.songs if s.mood}),
                )
            except Exception as exc:
                self.last_transient = True
                self.last_note = (f"Couldn't interpret the request via Gemini "
                                  f"({type(exc).__name__}) — using rule-based guesses.")
            if phrase:
                candidates = self.retriever.retrieve(phrase)
                if candidates:
                    self.last_interpretation = phrase
                else:
                    self.last_note = (f'Interpreted your request as "{phrase}" but '
                                      f"nothing in the catalog matches — showing "
                                      f"rule-based guesses.")

        self.last_candidates = candidates

        if candidates and self.generator.is_available():
            recs = self._try_generate(query, candidates, k)
            if recs is not None:
                if self.last_interpretation and not self.last_note:
                    self.last_note = (f'No direct matches — interpreted your request '
                                      f'as "{self.last_interpretation}".')
                return recs
        elif not self.generator.is_available() and candidates:
            self.last_note = "No GEMINI_API_KEY set — using rule-based fallback."

        return self._use_fallback(query, candidates, k)

    def recommend_similar(self, playlist: List[Song], k: int = 5) -> List[Recommendation]:
        """Suggest catalog songs that fit an existing playlist.

        Retrieval = centroid similarity over the index (never returns songs
        already in the playlist); generation/validation/fallback work exactly
        like the query flow.
        """
        self.last_interpretation = ""
        self.last_transient = False
        self.last_note = ""
        candidates = self.retriever.index.similar(playlist, top_n=max(8, k + 3))
        self.last_candidates = candidates
        if not candidates:
            self.last_source = "fallback"
            self.last_note = "Couldn't find anything similar in the catalog."
            return []

        if self.generator.is_available():
            query = self._playlist_query(playlist)
            recs = self._try_generate(query, candidates, k)
            if recs is not None:
                return recs
        else:
            self.last_note = "No GEMINI_API_KEY set — using similarity ranking only."

        self.last_source = "fallback"
        return [
            Recommendation(
                song=c.song, rank=i,
                explanation=("similar to your playlist — shares: "
                             + ", ".join(c.matched_terms)
                             if c.matched_terms else "similar overall vibe"),
                source="fallback",
            )
            for i, c in enumerate(candidates[:k], start=1)
        ]

    @staticmethod
    def _playlist_query(playlist: List[Song]) -> str:
        parts = []
        for s in playlist[:10]:
            desc = f'"{s.title}" by {s.artist}'
            if s.genre:
                desc += f" ({s.genre})"
            parts.append(desc)
        return ("songs that fit a playlist containing: " + "; ".join(parts)
                + ". Pick additions that match the playlist's overall style.")

    def _genre_vocabulary(self, limit: int = 40) -> List[str]:
        """Most common individual genres (multi-genre cells are split), so the
        interpretation prompt stays small even for large catalogs."""
        counts: dict = {}
        for song in self.retriever.index.songs:
            for genre in song.genre.split(","):
                genre = genre.strip()
                if genre:
                    counts[genre] = counts.get(genre, 0) + 1
        ranked = sorted(counts, key=counts.get, reverse=True)
        return ranked[:limit]

    def beyond_catalog(self, query: str, k: int = 3) -> List[dict]:
        """Optional, clearly-labeled extra: real-world song ideas from the
        LLM's general knowledge. NOT grounded in the catalog and NOT
        validated — kept separate from the RAG Recommendation flow."""
        if not self.generator.is_available():
            return []
        try:
            return self.generator.suggest_beyond_catalog(query, k)
        except Exception:
            return []

    def _try_generate(self, query: str, candidates: List[Candidate],
                      k: int) -> Optional[List[Recommendation]]:
        """Generate with Gemini; retry once on validation failure."""
        song_by_title = {c.song.title: c.song for c in candidates}
        correction = ""
        for attempt in range(2):
            try:
                picks = self.generator.generate(query, candidates, k, correction)
            except Exception as exc:  # network/auth/JSON errors -> fallback
                self.last_transient = True
                self.last_note = f"Gemini call failed ({type(exc).__name__}) — used fallback."
                return None

            picks = self.validator.deduplicate(picks, candidates)
            hallucinated = self.validator.find_hallucinated_titles(picks, candidates)
            if picks and not hallucinated:
                self.last_source = "gemini"
                if attempt == 1:
                    self.last_note = "First response had invalid titles; retry succeeded."
                else:
                    self.last_note = ""
                return [
                    Recommendation(
                        song=song_by_title[self.validator.match_title(p["title"], candidates)],
                        rank=i,
                        explanation=p["reason"] or "recommended by Gemini",
                        source="gemini")
                    for i, p in enumerate(picks, start=1)
                ]
            correction = (
                "Your previous answer included titles not in the candidate list: "
                + ", ".join(hallucinated or ["<empty response>"])
                + ". Answer again using ONLY exact titles from the numbered list."
            )
        self.last_transient = True
        self.last_note = "Gemini kept returning invalid titles — used fallback."
        return None

    def _use_fallback(self, query: str, candidates: List[Candidate],
                      k: int) -> List[Recommendation]:
        picks = self.fallback.recommend(query, candidates, k)
        self.last_source = "fallback"
        if not candidates and not self.last_note:
            self.last_note = ("No strong keyword matches in the catalog — "
                              "showing best rule-based guesses.")
        return [
            Recommendation(song=p["song"], rank=i, explanation=p["reason"],
                           source="fallback")
            for i, p in enumerate(picks, start=1)
        ]
