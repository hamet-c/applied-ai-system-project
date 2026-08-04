"""Song data model and CSV catalog loading."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Project root (one level above src/), so the catalog loads regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "songs.csv"


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


@dataclass
class UserProfile:
    """Represents a user's taste preferences (legacy profile format)."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


class SongCatalog:
    """Loads and holds the song catalog from a CSV file."""

    def __init__(self, csv_path: str = None):
        self.csv_path = str(csv_path or DEFAULT_CSV_PATH)

    def load_songs(self) -> List[Song]:
        songs: List[Song] = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append(
                    Song(
                        id=int(row["id"]),
                        title=row["title"],
                        artist=row["artist"],
                        genre=row["genre"],
                        mood=row["mood"],
                        energy=float(row["energy"]),
                        tempo_bpm=float(row["tempo_bpm"]),
                        valence=float(row["valence"]),
                        danceability=float(row["danceability"]),
                        acousticness=float(row["acousticness"]),
                    )
                )
        return songs
