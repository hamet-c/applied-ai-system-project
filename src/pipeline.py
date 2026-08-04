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
        if candidates is None:
            candidates = self.retriever.retrieve(query)

        if candidates and self.generator.is_available():
            recs = self._try_generate(query, candidates, k)
            if recs is not None:
                return recs
        elif not self.generator.is_available():
            self.last_note = "No GEMINI_API_KEY set — using rule-based fallback."

        return self._use_fallback(query, candidates, k)

    def _try_generate(self, query: str, candidates: List[Candidate],
                      k: int) -> Optional[List[Recommendation]]:
        """Generate with Gemini; retry once on validation failure."""
        by_title = {c.song.title.lower(): c.song for c in candidates}
        correction = ""
        for attempt in range(2):
            try:
                picks = self.generator.generate(query, candidates, k, correction)
            except Exception as exc:  # network/auth/JSON errors -> fallback
                self.last_note = f"Gemini call failed ({type(exc).__name__}) — used fallback."
                return None

            picks = self.validator.deduplicate(picks)
            hallucinated = self.validator.find_hallucinated_titles(picks, candidates)
            if picks and not hallucinated:
                self.last_source = "gemini"
                if attempt == 1:
                    self.last_note = "First response had invalid titles; retry succeeded."
                else:
                    self.last_note = ""
                return [
                    Recommendation(song=by_title[p["title"].lower()], rank=i,
                                   explanation=p["reason"] or "recommended by Gemini",
                                   source="gemini")
                    for i, p in enumerate(picks, start=1)
                ]
            correction = (
                "Your previous answer included titles not in the candidate list: "
                + ", ".join(hallucinated or ["<empty response>"])
                + ". Answer again using ONLY exact titles from the numbered list."
            )
        self.last_note = "Gemini kept returning invalid titles — used fallback."
        return None

    def _use_fallback(self, query: str, candidates: List[Candidate],
                      k: int) -> List[Recommendation]:
        picks = self.fallback.recommend(query, candidates, k)
        self.last_source = "fallback"
        if not candidates:
            self.last_note = ("No strong keyword matches in the catalog — "
                              "showing best rule-based guesses.")
        return [
            Recommendation(song=p["song"], rank=i, explanation=p["reason"],
                           source="fallback")
            for i, p in enumerate(picks, start=1)
        ]
