"""Grounded LLM generation: the G in RAG.

GeminiGenerator hands the user's request plus ONLY the retrieved candidate
songs to the Gemini API and asks for ranked picks with reasons, returned as
strict JSON. It never sees the full catalog — grounding happens by
construction, and the Validator double-checks the output anyway.
"""

import json
import os
from typing import List, Optional

from src.index import Candidate

DEFAULT_MODEL = "gemini-2.5-flash"


def _load_dotenv_if_present() -> None:
    """Load a local .env file so GEMINI_API_KEY can live outside the repo."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


class GeminiGenerator:
    """Calls Gemini to rank and explain picks, grounded in the candidates."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        _load_dotenv_if_present()
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = None

    def is_available(self) -> bool:
        """True when we have an API key and the google-genai package."""
        if not self.api_key:
            return False
        try:
            import google.genai  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def build_grounded_prompt(self, query: str, candidates: List[Candidate],
                              k: int, correction: str = "") -> str:
        lines = []
        for i, cand in enumerate(candidates, start=1):
            s = cand.song
            lines.append(
                f'{i}. "{s.title}" by {s.artist} — genre: {s.genre}, mood: {s.mood}, '
                f"energy: {s.energy}, tempo: {s.tempo_bpm:.0f} BPM, valence: {s.valence}, "
                f"danceability: {s.danceability}, acousticness: {s.acousticness}"
            )
        candidate_block = "\n".join(lines)

        prompt = f"""You are the recommendation engine of a small music app.

A user asked for: "{query}"

These are the ONLY songs you may recommend (retrieved from the catalog):
{candidate_block}

Pick the {k} best songs for this request, best first.

Rules:
- Recommend ONLY songs from the list above, copying their titles EXACTLY.
- Never invent a song. If nothing fits well, pick the closest matches anyway.
- Each reason must be one sentence tied to the user's request, mentioning
  concrete attributes (genre, mood, energy, tempo, acousticness).
- No duplicate songs.

Respond with JSON only, exactly in this shape:
{{"picks": [{{"title": "<exact title>", "reason": "<one sentence>"}}]}}"""
        if correction:
            prompt += f"\n\nIMPORTANT CORRECTION: {correction}"
        return prompt

    def generate(self, query: str, candidates: List[Candidate], k: int = 5,
                 correction: str = "") -> List[dict]:
        """Return a list of {"title": ..., "reason": ...} picks from Gemini."""
        from google.genai import types

        client = self._get_client()
        prompt = self.build_grounded_prompt(query, candidates, k, correction)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        data = json.loads(response.text)
        picks = data.get("picks", [])
        if not isinstance(picks, list):
            raise ValueError("Gemini response JSON missing 'picks' list")
        return [
            {"title": str(p.get("title", "")).strip(),
             "reason": str(p.get("reason", "")).strip()}
            for p in picks
            if isinstance(p, dict) and p.get("title")
        ][:k]
