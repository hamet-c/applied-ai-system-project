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

    _QUOTES = "\"'“”‘’"

    @staticmethod
    def _normalize(title: str) -> str:
        return title.strip().lower()

    def match_title(self, raw_title: str, candidates: List[Candidate]) -> str:
        """Map a model-returned title to the real candidate title, or ''.

        Models sometimes echo decoration from the prompt ('\"Moonlight\" by
        XXXTENTACION'). Each relaxation (strip quotes, drop a trailing
        'by <artist>') is only accepted if the result IS a candidate title,
        so hallucinated songs still never match.
        """
        allowed = {self._normalize(c.song.title): c.song.title for c in candidates}
        title = self._normalize(raw_title)
        if title in allowed:
            return allowed[title]
        unquoted = title.strip(self._QUOTES).strip()
        if unquoted in allowed:
            return allowed[unquoted]
        if " by " in unquoted:
            base = unquoted.rsplit(" by ", 1)[0].strip().strip(self._QUOTES).strip()
            if base in allowed:
                return allowed[base]
        return ""

    def find_hallucinated_titles(self, picks: List[dict],
                                 candidates: List[Candidate]) -> List[str]:
        """Titles the model returned that are NOT in the retrieved set."""
        return [p["title"] for p in picks
                if not self.match_title(p["title"], candidates)]

    def deduplicate(self, picks: List[dict],
                    candidates: List[Candidate] = None) -> List[dict]:
        seen = set()
        unique = []
        for p in picks:
            key = (self.match_title(p["title"], candidates) or p["title"].strip()).lower() \
                if candidates else self._normalize(p["title"])
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
