"""Retriever: the R in RAG. Finds candidate songs for a free-text query."""

from typing import List

from src.catalog import Song
from src.index import Candidate, SongIndex


class Retriever:
    """Matches a query against the SongIndex and returns top-N candidates."""

    def __init__(self, songs: List[Song], top_n: int = 8):
        self.top_n = top_n
        self.index = SongIndex()
        self.index.build(songs)

    def retrieve(self, query: str) -> List[Candidate]:
        return self.index.search(query, top_n=self.top_n)
