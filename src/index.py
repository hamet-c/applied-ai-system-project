"""Searchable text index over the song catalog.

Each song is turned into a small text "document": its title, artist, genre
and mood, plus its numeric features translated into descriptive words
(energy 0.9 -> "energetic high-energy loud ...") and derived context tags
(low energy + acoustic -> "study focus ..."). Free-text queries are matched
against these documents with TF-IDF + cosine similarity.

Important tokens (genre, mood, context tags) are repeated in the document,
which acts as a field boost: a genre word in the query counts more than an
incidental word. This is the same trick BM25F-style search engines use.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.catalog import Song

# Words that carry no meaning for matching ("play some songs for me").
STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your", "it",
    "is", "are", "was", "be", "been", "and", "or", "but", "so", "that",
    "this", "these", "those", "of", "to", "for", "with", "in", "on", "at",
    "by", "from", "as", "some", "any", "want", "wants", "need", "needs",
    "give", "play", "playing", "find", "get", "like", "really", "very",
    "please", "something", "sometimes", "bit", "little", "lot", "kind",
    "kinda", "sort", "make", "put", "song", "songs", "music", "track",
    "tracks", "tune", "tunes", "playlist", "listen", "listening", "hear",
    "vibe", "vibes", "feel", "feeling", "mood", "good", "great", "nice",
    "best", "top",
}

# Genres that real listeners treat as neighbors. A metal query should give
# some credit to rock, etc. (This directly targets the "exact genre match
# only" bias documented in the v1 model card.)
RELATED_GENRES = {
    "rock": ["metal", "punk"],
    "metal": ["rock"],
    "pop": ["indie pop", "dream pop", "synthwave"],
    "indie pop": ["pop", "dream pop"],
    "dream pop": ["pop", "indie pop", "ambient"],
    "synthwave": ["electronic", "pop"],
    "electronic": ["synthwave", "pop"],
    "lofi": ["ambient", "jazz"],
    "ambient": ["lofi", "classical"],
    "classical": ["ambient"],
    "jazz": ["r&b", "lofi"],
    "r&b": ["jazz"],
    "folk": ["indie pop"],
    "hip-hop": ["r&b"],
    "reggae": ["folk"],
}


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-letters, drop stopwords, strip simple plurals."""
    raw = re.findall(r"[a-z0-9&]+", text.lower())
    tokens = []
    for tok in raw:
        if tok in STOPWORDS:
            continue
        # Tiny stemmer: "beats" -> "beat" so plural queries still match.
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


def _energy_words(energy: float) -> str:
    if energy < 0.35:
        return "calm mellow soft quiet gentle peaceful low-energy relaxing soothing"
    if energy > 0.65:
        return "energetic high-energy loud powerful upbeat lively pumped hype"
    return "moderate balanced mid-energy easygoing"


def _valence_words(valence: float) -> str:
    if valence < 0.4:
        return "sad melancholy dark somber emotional brooding moody"
    if valence > 0.65:
        return "happy cheerful positive uplifting feel-good bright sunny joyful"
    return ""


def _context_tags(song: Song) -> str:
    """Derived 'editorial' tags, like the activity playlists real apps curate."""
    tags = []
    study_markers = ("lofi", "lo-fi", "ambient", "classical", "sleep", "study", "chillhop")
    if ((song.energy < 0.5 and song.acousticness > 0.5)
            or any(m in song.genre for m in study_markers)):
        tags.append("study studying focus focused concentration homework reading working coding background")
    if song.energy > 0.8 and song.danceability > 0.6:
        tags.append("workout gym running exercise training pump adrenaline")
    if song.danceability > 0.75 and song.energy > 0.7:
        tags.append("party dance dancing celebration club night-out")
    if song.energy < 0.45:
        tags.append("chill chilled relax relaxed relaxing laid-back unwind wind-down late-night")
    if song.energy < 0.3:
        # "asleep"/"sleepy" included: edge-case testing showed the query
        # "music to fall asleep to" missed the plain "sleep" vocabulary.
        tags.append("sleep sleeping asleep sleepy bedtime nap meditation quiet-night")
    if "synthwave" in song.genre:
        tags.append("driving drive night-drive road roadtrip retro neon")
    return " ".join(tags)


@dataclass
class Candidate:
    """A retrieved song plus why the retriever surfaced it."""
    song: Song
    relevance_score: float
    matched_terms: List[str] = field(default_factory=list)


class SongIndex:
    """TF-IDF index over song documents, searched with cosine similarity."""

    def __init__(self):
        self.songs: List[Song] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_vectors: List[Dict[str, float]] = []
        self._idf: Dict[str, float] = {}

    def to_document(self, song: Song) -> str:
        """Render one song as searchable text. Repetition = importance.

        Datasets without audio features load them as a neutral 0.5 — those
        songs get no feature words rather than misleading ones. Multi-genre
        strings ("country hip hop, southern hip hop") match related genres
        by substring.
        """
        related = set()
        for key, values in RELATED_GENRES.items():
            if key and key in song.genre:
                related.update(values)
        genre_extra = " ".join(sorted(related))

        tempo_words = ""
        if song.tempo_bpm:
            tempo_words = "slow" if song.tempo_bpm < 85 else ("fast quick" if song.tempo_bpm > 125 else "")
        acoustic_words = ""
        if song.acousticness > 0.6:
            acoustic_words = "acoustic organic unplugged natural instrumental"
        elif song.acousticness < 0.25:
            acoustic_words = "electronic electric synth produced"
        dance_words = "danceable groovy bouncy rhythmic" if song.danceability > 0.7 else ""
        energy_words = _energy_words(song.energy) if song.energy != 0.5 else ""
        valence_words = _valence_words(song.valence) if song.valence != 0.5 else ""
        popular_words = "popular hit well-known famous" if song.popularity >= 75 else ""

        parts = [
            song.title,
            song.artist,
            (song.genre + " ") * 3,          # genre: strongest field
            (song.mood + " ") * 3,           # mood: equally explicit intent
            genre_extra,                      # related genres: partial credit
            (_context_tags(song) + " ") * 2,  # activity intent ("studying")
            energy_words,
            valence_words,
            acoustic_words,
            dance_words,
            tempo_words,
            popular_words,
            "explicit" if song.explicit else "",
        ]
        return " ".join(p for p in parts if p)

    def build(self, songs: List[Song]) -> None:
        self.songs = list(songs)
        self._doc_tokens = [tokenize(self.to_document(s)) for s in self.songs]

        # Inverse document frequency: rare terms are more informative.
        n_docs = len(self._doc_tokens)
        df: Dict[str, int] = {}
        for tokens in self._doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._idf = {
            term: math.log((n_docs + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

        # Precompute normalized TF-IDF vectors for every document.
        self._doc_vectors = []
        for tokens in self._doc_tokens:
            vec = self._vectorize(tokens)
            self._doc_vectors.append(vec)

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        vec = {
            term: count * self._idf.get(term, 1.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {term: w / norm for term, w in vec.items()}

    def similar(self, seed_songs: List[Song], top_n: int = 8) -> List[Candidate]:
        """Songs most similar to a set of seeds (e.g. a playlist).

        Averages the seeds' TF-IDF vectors into a 'taste centroid' and ranks
        every other song by cosine similarity to it — pure content-based
        recommendation over the same index that search uses.
        """
        seed_ids = {s.id for s in seed_songs}
        seed_vecs = [vec for song, vec in zip(self.songs, self._doc_vectors)
                     if song.id in seed_ids]
        if not seed_vecs:
            return []

        centroid: Dict[str, float] = {}
        for vec in seed_vecs:
            for term, w in vec.items():
                centroid[term] = centroid.get(term, 0.0) + w / len(seed_vecs)
        # The centroid's heaviest terms describe the playlist's shared vibe.
        top_terms = sorted(centroid, key=centroid.get, reverse=True)[:15]

        candidates: List[Candidate] = []
        for song, doc_tokens, doc_vec in zip(self.songs, self._doc_tokens, self._doc_vectors):
            if song.id in seed_ids:
                continue
            score = sum(w * doc_vec.get(term, 0.0) for term, w in centroid.items())
            if score <= 0:
                continue
            score *= 1.0 + 0.15 * (song.popularity / 100.0)
            matched = [t for t in top_terms if t in doc_tokens][:5]
            candidates.append(Candidate(song=song, relevance_score=score, matched_terms=matched))

        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:top_n]

    def search(self, query: str, top_n: int = 8) -> List[Candidate]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.songs:
            return []
        query_vec = self._vectorize(query_tokens)

        candidates: List[Candidate] = []
        for song, doc_tokens, doc_vec in zip(self.songs, self._doc_tokens, self._doc_vectors):
            score = sum(w * doc_vec.get(term, 0.0) for term, w in query_vec.items())
            if score <= 0:
                continue
            # Mild popularity boost: among similar matches, well-known first.
            score *= 1.0 + 0.15 * (song.popularity / 100.0)
            matched = sorted(set(query_tokens) & set(doc_tokens))
            candidates.append(Candidate(song=song, relevance_score=score, matched_terms=matched))

        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        return candidates[:top_n]
