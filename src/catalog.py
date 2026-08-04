"""Song data model and schema-adaptive CSV catalog loading.

Supports three CSV shapes, detected from the header:
  1. Project/fixture schema:  title, artist, genre, mood + audio features
  2. Spotify artist-genres:   track_name, artist_name, artist_genres,
                              track_popularity, explicit (no audio features)
  3. Spotify audio-features:  name, artists, valence, energy, tempo...
                              (features but no genre)

Missing values degrade gracefully: absent audio features become a neutral
0.5, mood is derived from valence/energy (or genre keywords) when there is
no mood column, and an absent genre becomes an empty string.
"""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Project root (one level above src/), so the catalog loads regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "songs.csv"

NEUTRAL = 0.5  # value for audio features the dataset doesn't provide


@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: float = 0.0  # 0-100 when the dataset provides it
    explicit: bool = False


@dataclass
class UserProfile:
    """Represents a user's taste preferences (legacy profile format)."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def derive_mood(valence: float, energy: float) -> str:
    """Rough mood from the valence/energy quadrant (used when no mood column)."""
    if valence >= 0.6 and energy >= 0.6:
        return "happy"
    if valence >= 0.6:
        return "relaxed"
    if valence < 0.4 and energy >= 0.7:
        return "intense"
    if valence < 0.4 and energy < 0.4:
        return "sad"
    return "moody"


def mood_from_genres(genres: str) -> str:
    """Best-effort mood from genre keywords (used when no audio features)."""
    g = genres.lower()
    for keyword, mood in (
        ("metal", "intense"), ("hardcore", "intense"), ("punk", "intense"),
        ("emo", "sad"), ("sad", "sad"), ("slowcore", "sad"),
        ("lo-fi", "chill"), ("lofi", "chill"), ("chill", "chill"),
        ("ambient", "chill"), ("sleep", "chill"),
        ("classical", "relaxed"), ("jazz", "relaxed"), ("acoustic", "relaxed"),
        ("dance", "energetic"), ("edm", "energetic"), ("house", "energetic"),
        ("party", "energetic"),
    ):
        if keyword in g:
            return mood
    return ""


def _parse_artists(raw: str) -> str:
    """'["A", "B"]' -> 'A, B'; plain strings pass through."""
    raw = (raw or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return ", ".join(str(a) for a in parsed)
        except (ValueError, SyntaxError):
            pass
    return raw


def _f(row: Dict, key: str, default: float = NEUTRAL) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


class SongCatalog:
    """Loads and holds the song catalog from a CSV file, any known schema."""

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = str(csv_path or DEFAULT_CSV_PATH)

    def load_songs(self) -> List[Song]:
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if "title" in fields:
                parse = self._parse_project_row
            elif "track_name" in fields:
                parse = self._parse_artist_genres_row
            elif "name" in fields:
                parse = self._parse_audio_features_row
            else:
                raise ValueError(
                    f"Unrecognized songs CSV schema: {sorted(fields)[:10]}"
                )

            songs: List[Song] = []
            seen = set()
            for i, row in enumerate(reader, start=1):
                song = parse(i, row)
                if song is None:
                    continue
                # Dedupe identical title+artist (albums vs singles etc.).
                key = (song.title.lower(), song.artist.lower())
                if key in seen:
                    continue
                seen.add(key)
                songs.append(song)
        return songs

    @staticmethod
    def _parse_project_row(i: int, row: Dict) -> Optional[Song]:
        return Song(
            id=int(row.get("id", i) or i),
            title=row["title"],
            artist=row.get("artist", ""),
            genre=row.get("genre", ""),
            mood=row.get("mood", ""),
            energy=_f(row, "energy"),
            tempo_bpm=_f(row, "tempo_bpm", 0.0),
            valence=_f(row, "valence"),
            danceability=_f(row, "danceability"),
            acousticness=_f(row, "acousticness"),
            popularity=_f(row, "popularity", 0.0),
        )

    @staticmethod
    def _parse_artist_genres_row(i: int, row: Dict) -> Optional[Song]:
        title = (row.get("track_name") or "").strip()
        if not title:
            return None
        genres = (row.get("artist_genres") or "").strip()
        if genres.upper() == "N/A":
            genres = ""
        return Song(
            id=i,
            title=title,
            artist=(row.get("artist_name") or "").strip(),
            genre=genres.lower(),
            mood=mood_from_genres(genres),
            energy=NEUTRAL,
            tempo_bpm=0.0,
            valence=NEUTRAL,
            danceability=NEUTRAL,
            acousticness=NEUTRAL,
            popularity=_f(row, "track_popularity", 0.0),
            explicit=(row.get("explicit", "").strip().upper() == "TRUE"),
        )

    @staticmethod
    def _parse_audio_features_row(i: int, row: Dict) -> Optional[Song]:
        title = (row.get("name") or "").strip()
        if not title:
            return None
        energy = _f(row, "energy")
        valence = _f(row, "valence")
        return Song(
            id=i,
            title=title,
            artist=_parse_artists(row.get("artists", "")),
            genre=(row.get("track_genre") or "").strip().lower(),
            mood=derive_mood(valence, energy),
            energy=energy,
            tempo_bpm=_f(row, "tempo", 0.0),
            valence=valence,
            danceability=_f(row, "danceability"),
            acousticness=_f(row, "acousticness"),
            popularity=_f(row, "popularity", 0.0),
            explicit=str(row.get("explicit", "")).strip().lower() in ("1", "true"),
        )
