"""Runtime output checking: rejects hallucinated or duplicated picks.

This is the first of the system's three checks on AI output (the other two
are the EvaluationHarness and the human reviewing the retrieved panel).
An LLM asked to "only use these songs" will still occasionally invent a
title — the Validator catches that before it ever reaches the user.
"""

from typing import List

from src.index import Candidate


class Validator:
    """Checks LLM picks against the retrieved candidate set."""

    @staticmethod
    def _normalize(title: str) -> str:
        return title.strip().lower()

    def find_hallucinated_titles(self, picks: List[dict],
                                 candidates: List[Candidate]) -> List[str]:
        """Titles the model returned that are NOT in the retrieved set."""
        allowed = {self._normalize(c.song.title) for c in candidates}
        return [p["title"] for p in picks
                if self._normalize(p["title"]) not in allowed]

    def deduplicate(self, picks: List[dict]) -> List[dict]:
        seen = set()
        unique = []
        for p in picks:
            key = self._normalize(p["title"])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def validate(self, picks: List[dict], candidates: List[Candidate]) -> bool:
        """True when every pick is a real, non-duplicated retrieved song."""
        if not picks:
            return False
        if len(self.deduplicate(picks)) != len(picks):
            return False
        return not self.find_hallucinated_titles(picks, candidates)
