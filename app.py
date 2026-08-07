"""Songly — Streamlit UI for the RAG music recommender.

Run with:  streamlit run app.py
"""

import streamlit as st

from src import democache
from src.pipeline import RecommenderPipeline

st.set_page_config(page_title="Songly", page_icon="🎵", layout="centered")

EXAMPLE_QUERIES = [
    "calm acoustic songs for studying late at night",
    "high energy workout music",
    "something happy for a sunny morning drive",
]


@st.cache_resource
def get_pipeline() -> RecommenderPipeline:
    return RecommenderPipeline.from_catalog()


def get_recommendations(query: str, k: int):
    """Cached per (query, k) so reruns don't re-call the Gemini API.

    Results caused by a transient Gemini failure (rate limit, network) are
    deliberately NOT cached — otherwise one bad call would freeze that query
    in fallback mode for the whole session.
    """
    cache = st.session_state.setdefault("results_cache", {})
    key = (query, k)
    if key in cache:
        return cache[key]

    pipe = get_pipeline()
    # Disk cache (warmed by `python -m src.warmup`) survives restarts, so a
    # rate limit or dropped network can't break a demo of these queries.
    entry = democache.get(democache.query_key(query, k))
    if entry:
        restored = democache.decode_recommendations(entry, songs_by_id)
        if restored:
            cache[key] = restored
            return restored

    recommendations = pipe.recommend(query, k=k)
    result = (pipe.last_candidates, recommendations, pipe.last_source, pipe.last_note)
    if not pipe.last_transient:  # transient API failures retry next time
        cache[key] = result
    return result


def get_beyond_catalog(query: str):
    """Ungrounded real-song ideas from Gemini; cached only on success."""
    cache = st.session_state.setdefault("beyond_cache", {})
    if query in cache:
        return cache[query]
    entry = democache.get(democache.beyond_key(query))
    if entry and entry.get("suggestions"):
        cache[query] = entry["suggestions"]
        return cache[query]
    suggestions = get_pipeline().beyond_catalog(query, k=3)
    if suggestions:
        cache[query] = suggestions
    return suggestions


def get_similar(playlist_songs, k: int):
    """Playlist suggestions, disk-cached the same way as queries."""
    pipe = get_pipeline()
    entry = democache.get(democache.similar_key([s.id for s in playlist_songs], k))
    if entry:
        restored = democache.decode_recommendations(entry, songs_by_id)
        if restored:
            _, recs, source, note = restored
            return recs, source, note
    recs = pipe.recommend_similar(playlist_songs, k=k)
    return recs, pipe.last_source, pipe.last_note


pipeline = get_pipeline()
songs_by_id = {s.id: s for s in pipeline.retriever.index.songs}

if "playlist" not in st.session_state:
    st.session_state.playlist = []  # list of song ids, insertion order


def _add_to_playlist(song_id: int) -> None:
    if song_id not in st.session_state.playlist:
        st.session_state.playlist.append(song_id)


def _remove_from_playlist(song_id: int) -> None:
    st.session_state.playlist = [i for i in st.session_state.playlist if i != song_id]


def render_song_card(song, rank: int, explanation: str, key_prefix: str) -> None:
    """One recommendation card with an add-to-playlist button."""
    with st.container(border=True):
        st.markdown(f"### {rank}. {song.title}")
        tags = []
        if song.genre:
            tags.append(f"`{song.genre}`")
        if song.mood:
            tags.append(f"`{song.mood}`")
        if song.energy != 0.5:
            tags.append(f"`energy {song.energy:.2f}`")
        if song.tempo_bpm:
            tags.append(f"`{song.tempo_bpm:.0f} BPM`")
        if song.popularity:
            tags.append(f"`popularity {song.popularity:.0f}`")
        st.markdown(f"*{song.artist}* · " + " ".join(tags))
        st.markdown(f"💬 {explanation}")
        in_playlist = song.id in st.session_state.playlist
        st.button(
            "✅ In playlist" if in_playlist else "➕ Add to playlist",
            key=f"{key_prefix}_add_{song.id}",
            disabled=in_playlist,
            on_click=_add_to_playlist,
            args=(song.id,),
        )

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of picks", min_value=3, max_value=8, value=5)
    beyond = st.toggle(
        "🌍 Also suggest real songs (beyond the catalog)",
        value=True,
        help="Extra ideas from Gemini's general knowledge. These are NOT "
             "grounded in the catalog and NOT validated — shown separately "
             "from the RAG recommendations.",
        disabled=not pipeline.generator.is_available(),
    )

    st.divider()
    if pipeline.generator.is_available():
        st.success(f"Gemini connected\n\nmodel: `{pipeline.generator.model}`")
    else:
        st.warning(
            "No `GEMINI_API_KEY` found — running in **rule-based fallback "
            "mode** (templated explanations).\n\n"
            "Add a key to a `.env` file to enable LLM generation."
        )

    st.divider()
    st.subheader("How it works")
    st.markdown(
        "1. **Retrieve** — your words are matched against a searchable "
        "index of every song (TF-IDF).\n"
        "2. **Generate** — Gemini ranks and explains picks using *only* "
        "the retrieved songs.\n"
        "3. **Validate** — any invented title is rejected and retried.\n"
        "4. **Fallback** — no key or repeated failures? The original "
        "rule-based scorer takes over."
    )

# ---------- Main ----------
st.title("🎵 Songly")
st.caption(f"A RAG-powered music recommender over a "
           f"{len(pipeline.retriever.index.songs):,}-song catalog. "
           "Describe what you want to hear, in your own words.")


def _set_query(text: str) -> None:
    st.session_state.query = text


cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    col.button(example, on_click=_set_query, args=(example,), width='stretch')

query = st.text_input(
    "What are you in the mood for?",
    key="query",
    placeholder="e.g. sad slow songs for a rainy evening",
)

_catalog_songs = pipeline.retriever.index.songs
with st.expander(f"🎼 Browse the catalog ({len(_catalog_songs):,} songs)"):
    _browse = sorted(_catalog_songs, key=lambda s: s.popularity, reverse=True)[:500]
    if len(_catalog_songs) > len(_browse):
        st.caption(f"Showing the {len(_browse)} most popular of "
                   f"{len(_catalog_songs):,} songs. Recommendations can only "
                   "come from this catalog.")
    st.dataframe(
        [{"Title": s.title, "Artist": s.artist, "Genre": s.genre,
          "Popularity": int(s.popularity)}
         for s in _browse],
        width='stretch',
        hide_index=True,
    )

if query.strip():
    with st.spinner("Finding your songs..."):
        candidates, recommendations, source, note = get_recommendations(query.strip(), k)

    if not candidates:
        st.warning(
            f'Nothing in the catalog matches **"{query}"** — recommendations '
            "can only come from the songs in *Browse the catalog* above. The "
            "picks below are generic rule-based guesses. Try a genre, mood, "
            "activity, or an artist from the catalog."
        )

    if not recommendations:
        st.info("No recommendations — try different words.")
    else:
        badge = "🤖 Gemini (grounded generation)" if source == "gemini" else "🧮 Rule-based fallback"
        st.markdown(f"**Source:** {badge}")
        if note:
            st.caption(f"ℹ️ {note}")

        for rec in recommendations:
            render_song_card(rec.song, rec.rank, rec.explanation, "rec")

        if beyond:
            with st.spinner("Asking Gemini for ideas beyond the catalog..."):
                suggestions = get_beyond_catalog(query.strip())
            if suggestions:
                st.divider()
                st.markdown("#### 🌍 Beyond the catalog")
                st.caption(
                    "Real-world ideas from Gemini's general knowledge — **not** "
                    "retrieved from the catalog, **not** validated, and separate "
                    "from the grounded recommendations above. Titles may be "
                    "imperfect; treat these as starting points."
                )
                for s in suggestions:
                    st.markdown(f"- **{s['title']}** — {s['artist']}  \n  _{s['reason']}_")

        with st.expander("🔎 What was retrieved (RAG transparency)"):
            st.caption(
                "These are the only songs the generator was allowed to pick "
                "from. Compare them with the picks above to judge grounding."
            )
            st.dataframe(
                [
                    {
                        "Title": c.song.title,
                        "Artist": c.song.artist,
                        "Genre": c.song.genre,
                        "Mood": c.song.mood,
                        "Relevance": round(c.relevance_score, 3),
                        "Matched terms": ", ".join(c.matched_terms),
                    }
                    for c in candidates
                ],
                width='stretch',
                hide_index=True,
            )
else:
    st.info("Type a request above, or click an example to get started.")

# ---------- Playlist ----------
st.divider()
playlist_songs = [songs_by_id[i] for i in st.session_state.playlist if i in songs_by_id]
st.markdown(f"## 🎧 Your playlist ({len(playlist_songs)})")

snapshot = tuple(st.session_state.playlist)

if not playlist_songs:
    st.caption("Empty — use **➕ Add to playlist** on any recommendation, "
               "then get suggestions that fit what you've collected.")
else:
    for song in playlist_songs:
        left, right = st.columns([8, 1])
        left.markdown(f"**{song.title}** — {song.artist}"
                      + (f" · `{song.genre}`" if song.genre else ""))
        right.button("✖", key=f"pl_rm_{song.id}", help="Remove",
                     on_click=_remove_from_playlist, args=(song.id,))

    col_a, col_b = st.columns([2, 1])
    find = col_a.button("🎯 Find similar songs", type="primary",
                        width='stretch')
    col_b.button("🗑️ Clear playlist", width='stretch',
                 on_click=lambda: st.session_state.playlist.clear())

    if find:
        with st.spinner("Finding songs that fit your playlist..."):
            st.session_state.similar_results = get_similar(playlist_songs, k)
        st.session_state.similar_snapshot = snapshot

    if "similar_results" in st.session_state:
        recs, source, note = st.session_state.similar_results
        st.markdown("#### Because of what's in your playlist:")
        if st.session_state.get("similar_snapshot") != snapshot:
            st.caption("↻ Playlist changed since these suggestions — click "
                       "**Find similar songs** to refresh.")
        badge = ("🤖 Gemini (grounded generation)" if source == "gemini"
                 else "🧮 Similarity ranking")
        st.markdown(f"**Source:** {badge}")
        if note:
            st.caption(f"ℹ️ {note}")
        if not recs:
            st.info("Nothing similar found — try adding a few more songs.")
        for rec in recs:
            render_song_card(rec.song, rec.rank, rec.explanation, "sim")
