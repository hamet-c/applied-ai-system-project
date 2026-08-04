"""VibeFinder 2.0 — Streamlit UI for the RAG music recommender.

Run with:  streamlit run app.py
"""

import streamlit as st

from src.pipeline import RecommenderPipeline

st.set_page_config(page_title="VibeFinder 2.0", page_icon="🎵", layout="centered")

EXAMPLE_QUERIES = [
    "calm acoustic songs for studying late at night",
    "high energy workout music",
    "something happy for a sunny morning drive",
]


@st.cache_resource
def get_pipeline() -> RecommenderPipeline:
    return RecommenderPipeline.from_catalog()


@st.cache_data(show_spinner=False)
def get_recommendations(query: str, k: int):
    """Cached per (query, k) so reruns don't re-call the Gemini API."""
    pipe = get_pipeline()
    candidates = pipe.retriever.retrieve(query)
    recommendations = pipe.recommend(query, k=k, candidates=candidates)
    return candidates, recommendations, pipe.last_source, pipe.last_note


pipeline = get_pipeline()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of picks", min_value=3, max_value=8, value=5)

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
st.title("🎵 VibeFinder 2.0")
st.caption("A RAG-powered music recommender over a tiny catalog. "
           "Describe what you want to hear — in your own words.")


def _set_query(text: str) -> None:
    st.session_state.query = text


cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    col.button(example, on_click=_set_query, args=(example,), use_container_width=True)

query = st.text_input(
    "What are you in the mood for?",
    key="query",
    placeholder="e.g. sad slow songs for a rainy evening",
)

if query.strip():
    with st.spinner("Finding your songs..."):
        candidates, recommendations, source, note = get_recommendations(query.strip(), k)

    if not recommendations:
        st.info("No recommendations — try different words.")
    else:
        badge = "🤖 Gemini (grounded generation)" if source == "gemini" else "🧮 Rule-based fallback"
        st.markdown(f"**Source:** {badge}")
        if note:
            st.caption(f"ℹ️ {note}")

        for rec in recommendations:
            with st.container(border=True):
                st.markdown(f"### {rec.rank}. {rec.song.title}")
                st.markdown(f"*{rec.song.artist}* · "
                            f"`{rec.song.genre}` `{rec.song.mood}` "
                            f"`energy {rec.song.energy:.2f}` "
                            f"`{rec.song.tempo_bpm:.0f} BPM`")
                st.markdown(f"💬 {rec.explanation}")

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
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("Type a request above, or click an example to get started.")
