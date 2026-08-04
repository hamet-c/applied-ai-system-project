"""Command line runner for the RAG music recommender.

Usage:
    python -m src.main "calm acoustic songs for studying late"
    python -m src.main                 # runs three demo queries
    python -m src.main -k 3 "workout"  # top 3 picks

The Streamlit UI (streamlit run app.py) is the main interface; this CLI is
handy for quick checks and for capturing text output for the README.
"""

import argparse

from src.pipeline import RecommenderPipeline

DEMO_QUERIES = [
    "calm acoustic songs for studying late at night",
    "high energy workout music",
    "something happy and upbeat for a sunny morning drive",
]


def run_query(pipeline: RecommenderPipeline, query: str, k: int) -> None:
    recommendations = pipeline.recommend(query, k=k)
    candidates = pipeline.last_candidates  # includes interpreted retrieval

    print()
    print("=" * 60)
    print(f'  Query: "{query}"')
    print("=" * 60)

    print(f"\n  Retrieved candidates (top {len(candidates)}):")
    for cand in candidates:
        matched = ", ".join(cand.matched_terms) or "-"
        print(f"    - {cand.song.title} ({cand.song.genre}/{cand.song.mood})"
              f"  relevance={cand.relevance_score:.2f}  matched: {matched}")

    source = pipeline.last_source or "-"
    print(f"\n  Recommendations  [source: {source}]")
    if pipeline.last_note:
        print(f"  note: {pipeline.last_note}")
    for rec in recommendations:
        print(f"\n  {rec.rank}. {rec.song.title} - {rec.song.artist}")
        print(f"     why: {rec.explanation}")
    print()


def run_similar(pipeline: RecommenderPipeline, titles: list, k: int) -> None:
    """Playlist mode: find songs by title, then suggest similar ones."""
    all_songs = pipeline.retriever.index.songs
    playlist = []
    for title in titles:
        match = next((s for s in all_songs if title.lower() in s.title.lower()), None)
        if match:
            playlist.append(match)
        else:
            print(f'  (no catalog song matching "{title}" - skipped)')
    if not playlist:
        print("No playlist songs found.")
        return

    print()
    print("=" * 60)
    print("  Playlist:")
    for s in playlist:
        print(f"    - {s.title} - {s.artist}" + (f"  ({s.genre})" if s.genre else ""))
    print("=" * 60)

    recommendations = pipeline.recommend_similar(playlist, k=k)
    print(f"\n  Similar songs  [source: {pipeline.last_source or '-'}]")
    if pipeline.last_note:
        print(f"  note: {pipeline.last_note}")
    for rec in recommendations:
        print(f"\n  {rec.rank}. {rec.song.title} - {rec.song.artist}"
              + (f"  ({rec.song.genre})" if rec.song.genre else ""))
        print(f"     why: {rec.explanation}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG music recommender CLI")
    parser.add_argument("query", nargs="*", help="free-text music request")
    parser.add_argument("-k", type=int, default=5, help="number of picks (default 5)")
    parser.add_argument("--like", action="append", default=[], metavar="TITLE",
                        help="playlist mode: a song title already in your playlist "
                             "(repeatable); suggests similar catalog songs")
    args = parser.parse_args()

    pipeline = RecommenderPipeline.from_catalog()
    print(f"Loaded songs: {len(pipeline.retriever.index.songs)}")
    if not pipeline.generator.is_available():
        print("(no GEMINI_API_KEY found - running in rule-based fallback mode)")

    if args.like:
        run_similar(pipeline, args.like, args.k)
        return

    queries = [" ".join(args.query)] if args.query else DEMO_QUERIES
    for query in queries:
        run_query(pipeline, query, args.k)


if __name__ == "__main__":
    main()
