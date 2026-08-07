"""Pre-run demo queries and cache the results, so a live demo can't fail.

    python -m src.warmup                       # warm the default demo set
    python -m src.warmup "your own query"      # warm specific queries

Run this shortly before presenting. Afterwards the app answers those exact
queries from disk with no API call, so a rate limit or dropped network
can't interrupt the demo. Anything you type that wasn't warmed still works
normally (live call, or rule-based fallback if the API is unavailable).
"""

import sys

from src import democache
from src.pipeline import RecommenderPipeline

DEMO_QUERIES = [
    "calm acoustic songs for studying late at night",
    "high energy workout music",
    "something happy for a sunny morning drive",
    "sad slow songs for a rainy evening",
]

# Playlists to warm for the "find similar" demo: lists of title fragments.
DEMO_PLAYLISTS = [
    ["SAD!", "Lucid Dreams"],
]

K_VALUES = [5]  # the app's default slider position


def warm_queries(pipeline, queries, k_values) -> int:
    warmed = 0
    for query in queries:
        for k in k_values:
            recs = pipeline.recommend(query, k=k)
            if pipeline.last_source != "gemini":
                print(f"  ! {query!r} (k={k}) -> {pipeline.last_source}"
                      f" | {pipeline.last_note or 'no note'}")
                continue
            democache.put(
                democache.query_key(query, k),
                democache.encode_recommendations(
                    pipeline.last_candidates, recs,
                    pipeline.last_source, pipeline.last_note),
            )
            warmed += 1
            print(f"  + {query!r} (k={k}) -> {len(recs)} picks")

            suggestions = pipeline.beyond_catalog(query, k=3)
            if suggestions:
                democache.put(democache.beyond_key(query),
                              {"suggestions": suggestions})
                print(f"    + beyond-catalog: {len(suggestions)} ideas")
    return warmed


def warm_playlists(pipeline, playlists, k_values) -> int:
    songs = pipeline.retriever.index.songs
    warmed = 0
    for fragments in playlists:
        playlist = []
        for fragment in fragments:
            match = next((s for s in songs if fragment.lower() in s.title.lower()), None)
            if match:
                playlist.append(match)
        if not playlist:
            print(f"  ! no catalog songs matched {fragments}")
            continue
        for k in k_values:
            recs = pipeline.recommend_similar(playlist, k=k)
            if pipeline.last_source != "gemini":
                print(f"  ! playlist {fragments} -> {pipeline.last_source}")
                continue
            democache.put(
                democache.similar_key([s.id for s in playlist], k),
                democache.encode_recommendations(
                    pipeline.last_candidates, recs,
                    pipeline.last_source, pipeline.last_note),
            )
            warmed += 1
            print(f"  + playlist {[s.title for s in playlist]} -> {len(recs)} picks")
    return warmed


def main() -> None:
    pipeline = RecommenderPipeline.from_catalog()
    if not pipeline.generator.is_available():
        print("No GEMINI_API_KEY found. Nothing to warm: the app will use the "
              "rule-based fallback, which needs no network and never fails.")
        return

    queries = sys.argv[1:] or DEMO_QUERIES
    print(f"Warming {len(queries)} queries against {len(pipeline.retriever.index.songs)} songs...")
    warmed = warm_queries(pipeline, queries, K_VALUES)

    if not sys.argv[1:]:
        print("Warming playlists...")
        warmed += warm_playlists(pipeline, DEMO_PLAYLISTS, K_VALUES)

    print(f"\nCached {warmed} responses -> {democache.CACHE_PATH.name}")
    if warmed:
        print("These now answer instantly with no API call. Delete "
              f"{democache.CACHE_PATH.name} to force live calls again.")


if __name__ == "__main__":
    main()
