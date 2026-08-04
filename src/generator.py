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

# The "-latest" alias tracks Google's current stable Flash model. A pinned
# version ("gemini-2.5-flash") 404'd for new API keys within months — the
# alias keeps the app working as Google rotates models.
DEFAULT_MODEL = "gemini-flash-latest"


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
            # Only mention attributes the dataset actually provides (missing
            # audio features load as a neutral 0.5 / tempo 0) — otherwise the
            # model cites meaningless numbers in its reasons.
            attrs = []
            if s.genre:
                attrs.append(f"genre: {s.genre}")
            if s.mood:
                attrs.append(f"mood: {s.mood}")
            if s.energy != 0.5:
                attrs.append(f"energy: {s.energy}")
            if s.tempo_bpm:
                attrs.append(f"tempo: {s.tempo_bpm:.0f} BPM")
            if s.valence != 0.5:
                attrs.append(f"valence: {s.valence}")
            if s.danceability != 0.5:
                attrs.append(f"danceability: {s.danceability}")
            if s.acousticness != 0.5:
                attrs.append(f"acousticness: {s.acousticness}")
            if s.popularity:
                attrs.append(f"popularity: {s.popularity:.0f}/100")
            lines.append(f'{i}. "{s.title}" by {s.artist} — ' + ", ".join(attrs))
        candidate_block = "\n".join(lines)

        prompt = f"""You are the recommendation engine of a small music app.

A user asked for: "{query}"

These are the ONLY songs you may recommend (retrieved from the catalog):
{candidate_block}

Pick the {k} best songs for this request, best first.

Rules:
- Recommend ONLY songs from the list above, copying their titles EXACTLY.
- The "title" field must contain ONLY the song title — no quotes around it,
  no artist name, no "by ...".
- Never invent a song. If nothing fits well, pick the closest matches anyway.
- Each reason must be one sentence tied to the user's request, mentioning
  concrete attributes (genre, mood, energy, tempo, acousticness).
- No duplicate songs.

Respond with JSON only, exactly in this shape:
{{"picks": [{{"title": "<exact title>", "reason": "<one sentence>"}}]}}"""
        if correction:
            prompt += f"\n\nIMPORTANT CORRECTION: {correction}"
        return prompt

    def _call_json(self, prompt: str) -> dict:
        from google.genai import types

        client = self._get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.4,
            ),
        )
        return json.loads(response.text)

    def interpret_query(self, query: str, genres: List[str], moods: List[str]) -> str:
        """Translate an out-of-catalog request (e.g. a real artist) into
        catalog vocabulary, so retrieval can find the closest vibe."""
        prompt = f"""A music app user asked for: "{query}"

The app's catalog found no direct matches. Rewrite the request as a short
search phrase describing the MUSICAL STYLE the user wants, using only:
- genres from this list: {", ".join(sorted(genres))}
- moods from this list: {", ".join(sorted(moods))}
- energy words: calm, moderate, energetic
- optionally: acoustic or electronic

If the request names a real artist, describe that artist's typical style
with those words. Respond with JSON only: {{"search_phrase": "<phrase>"}}"""
        data = self._call_json(prompt)
        return str(data.get("search_phrase", "")).strip()

    def suggest_beyond_catalog(self, query: str, k: int = 3) -> List[dict]:
        """Real-world song suggestions from the model's general knowledge.
        NOT grounded, NOT validated — callers must label them as such."""
        prompt = f"""Suggest {k} real, well-known songs for this request: "{query}"

For each, give a one-sentence reason tied to the request.
Respond with JSON only, exactly in this shape:
{{"suggestions": [{{"title": "...", "artist": "...", "reason": "..."}}]}}"""
        data = self._call_json(prompt)
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            return []
        return [
            {"title": str(s.get("title", "")).strip(),
             "artist": str(s.get("artist", "")).strip(),
             "reason": str(s.get("reason", "")).strip()}
            for s in suggestions
            if isinstance(s, dict) and s.get("title")
        ][:k]

    def generate(self, query: str, candidates: List[Candidate], k: int = 5,
                 correction: str = "") -> List[dict]:
        """Return a list of {"title": ..., "reason": ...} picks from Gemini."""
        prompt = self.build_grounded_prompt(query, candidates, k, correction)
        data = self._call_json(prompt)
        picks = data.get("picks", [])
        if not isinstance(picks, list):
            raise ValueError("Gemini response JSON missing 'picks' list")
        return [
            {"title": str(p.get("title", "")).strip(),
             "reason": str(p.get("reason", "")).strip()}
            for p in picks
            if isinstance(p, dict) and p.get("title")
        ][:k]
