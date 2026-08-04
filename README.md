# 🎵 Songly: a RAG-Powered Music Recommender

## Original Project (Modules 1–3)

This project evolved from the **Music Recommender Simulation ("VibeFinder 1.0")**. The original version represented songs and a user taste profile as data, scored every song in an 18-track catalog with a hand-tuned weighted recipe (genre +2.0, mood +1.0, energy closeness up to +1.0, acousticness +0.5), and printed the top 5 picks with templated explanations. It only accepted rigid, pre-defined profiles (`genre=pop, mood=happy, energy=0.9`) and its evaluation exposed a clear bias: exact genre matches outweighed everything else, so a loud metal track ranked #1 for a user asking for *relaxed* music.

## Title and Summary

**Songly** turns that rule-based scorer into a **Retrieval-Augmented Generation (RAG)** system with a Streamlit web UI. You describe what you want to hear in your own words, like *"calm acoustic songs for studying late at night"*, and the system:

1. **Retrieves** the most relevant songs from the catalog using a TF-IDF search index,
2. **Generates** ranked picks with conversational explanations using the Gemini API, grounded in **only** the retrieved songs,
3. **Validates** the output, rejecting any hallucinated song title and retrying, and
4. **Falls back** to the original rule-based scorer when no API key is set or generation fails.

Why it matters: this is the same architecture real AI products use to keep LLMs honest. The model never answers from its imagination; it answers from retrieved data, and a validator plus an evaluation harness measure that it stays that way.

## Architecture Overview

The full UML diagram is in [diagrams/architecture.mmd](diagrams/architecture.mmd). In short:

```
User query → Retriever (TF-IDF over SongIndex) → Candidates
           → GeminiGenerator (grounded prompt: candidates ONLY)
           → Validator (hallucinated title? → retry once → fallback)
           → Recommendation[] → Streamlit UI
```

- **`SongCatalog`** ([src/catalog.py](src/catalog.py)): a schema-adaptive loader: it detects the CSV's shape and maps whatever columns exist. The shipped catalog is a real ~7,900-song Spotify dataset (artist genres + popularity); the original 18 fictional songs live on as a frozen test fixture. Missing audio features degrade gracefully (neutral values, no fabricated words), and mood is derived from genre keywords or valence/energy when absent.
- **`SongIndex`** ([src/index.py](src/index.py)): each song becomes a searchable text document: title, artist, genre(s), mood, plus available numeric features translated into words (energy 0.9 → "energetic, high-energy...") and derived activity tags. Important fields are repeated to act as a boost, related genres get partial credit (a metal query also surfaces rock), and popular songs get a mild ranking bump.
- **`Retriever`** ([src/retriever.py](src/retriever.py)): cosine similarity between the query and every song document; returns the top-8 candidates with their matched terms.
- **`GeminiGenerator`** ([src/generator.py](src/generator.py)): builds a grounded prompt containing *only* the candidates and asks Gemini for strict-JSON picks with one-sentence reasons.
- **`Validator`** ([src/validator.py](src/validator.py)): checks every returned title against the candidate set; hallucinations trigger one corrective retry, then fallback.
- **Query interpretation**: if retrieval finds *nothing* (say you search a real-world artist like "Drake"), Gemini translates the request into catalog vocabulary ("hip-hop, confident, energetic") and retrieval runs again. Grounding is preserved: the picks still come only from the catalog, and the note tells you how the request was interpreted.
- **Beyond the catalog (on by default, can be toggled off)**: real-world song ideas from Gemini's general knowledge, shown in a clearly-labeled separate section. These are deliberately *not* part of the RAG output: not retrieved, not validated, and marked as such in the UI.
- **Playlists + "find similar"**: add any recommended song to a playlist, then ask for songs that fit it. The playlist's TF-IDF vectors are averaged into a "taste centroid," every other song is ranked by cosine similarity to it (playlist members excluded), and Gemini ranks/explains the top candidates, with the same grounding and validation as the query flow. CLI: `python -m src.main --like "SAD!" --like "Lucid Dreams"`.
- **`FallbackScorer`** ([src/fallback.py](src/fallback.py)): the original v1 scoring recipe, now driven by a taste profile derived from the query text. Runs when there's no API key or generation fails.
- **`RecommenderPipeline`** ([src/pipeline.py](src/pipeline.py)): orchestrates the flow above; the CLI, UI, tests, and evaluator all share it.
- **`EvaluationHarness`** ([src/evaluation.py](src/evaluation.py)): golden-query testing (see Testing Summary).

Three checkpoints watch the AI's output: the **Validator** at runtime, the **EvaluationHarness** before changes ship, and the **human**: the UI shows a "What was retrieved" panel so you can compare the picks against the retrieval set and judge the grounding yourself.

## Setup Instructions

1. Clone the repo and create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional but recommended)* Enable Gemini generation: copy `.env.example` to `.env` and paste in a free API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey):

   ```
   GEMINI_API_KEY=your-key-here
   ```

   Without a key the app still works fully, using the rule-based fallback.

   > **Free-tier tip:** Google's free quotas are per-model and per-day. If the app starts falling back with a `429 RESOURCE_EXHAUSTED` note after heavy use, add `GEMINI_MODEL=gemini-flash-lite-latest` to your `.env`; the lite tier has a separate (larger) quota.

4. Run the web app:

   ```bash
   streamlit run app.py
   ```

   Or the CLI:

   ```bash
   python -m src.main "sad slow songs for a rainy evening"
   ```

5. Run the tests and the evaluation harness:

   ```bash
   pytest                      # 35 unit tests, all offline
   python -m src.evaluation    # golden-query metrics
   ```

## Sample Interactions

Real CLI output, captured from actual runs. Examples 1–4 were captured on the original 18-song fictional catalog (now the frozen test fixture); example 5 runs on the current real ~7,900-song Spotify catalog. Note for example 3: on the *real* catalog, Drake matches directly; interpretation now only kicks in for artists genuinely absent from the data.

**1. `python -m src.main "calm acoustic songs for studying late at night"`** (Gemini mode)

```
Retrieved candidates (top 8):
  - Spacewalk Thoughts (ambient/chill)      relevance=0.24  matched: acoustic, calm, late, night, studying
  - Autumn Nocturne (classical/melancholic) relevance=0.23  matched: acoustic, calm, late, night, studying
  - Midnight Coding (lofi/chill)            relevance=0.21  matched: acoustic, late, night, studying
  - Focus Flow (lofi/focused)               relevance=0.21  matched: acoustic, late, night, studying
  - Library Rain (lofi/chill)               relevance=0.21  matched: acoustic, late, night, studying
  ...

Recommendations  [source: gemini]
  1. Library Rain - Paper Lanterns
     why: With a high acousticness of 0.86 and a lofi genre, this song delivers a chill mood
          at a steady 72 BPM tempo with low 0.35 energy that is ideal for late-night study sessions.
  2. Spacewalk Thoughts - Orbit Bloom
     why: This ambient track features a very high acousticness of 0.92, a slow tempo of 60 BPM,
          a low 0.28 energy, and a chill mood perfect for keeping calm while working late.
  3. Coffee Shop Stories - Slow Stereo
     why: Offering a relaxed mood and smooth jazz genre, this track combines a high 0.89
          acousticness with a low 0.37 energy and a comfortable 90 BPM tempo for quiet focus.
  ...
```

Every pick is a real retrieved song, and every claim in the explanations (BPM, energy, acousticness) matches the catalog data. That's the grounding working.

**2. `python -m src.main "romantic r&b for a date night" -k 3`** (Gemini mode)

```
Retrieved candidates (top 8):
  - Velvet Hours (r&b/romantic)        relevance=0.73  matched: r&b, romantic
  - Concrete Kings (hip-hop/confident) relevance=0.10  matched: night, r&b
  - Coffee Shop Stories (jazz/relaxed) relevance=0.10  matched: night, r&b
  ...

Recommendations  [source: gemini]
  1. Velvet Hours - Mara Soul
     why: This track is a perfect fit for a date night with its r&b genre, romantic mood,
          moderate energy of 0.55, steady tempo of 88 BPM, and subtle acousticness of 0.31.
  2. Coffee Shop Stories - Slow Stereo
     why: This song provides a cozy date night vibe through its smooth jazz genre, relaxed
          mood, low energy of 0.37, gentle tempo of 90 BPM, and rich acousticness of 0.89.
  3. Spacewalk Thoughts - Orbit Bloom
     why: It offers an intimate date night atmosphere with its ambient genre, chill mood,
          low energy of 0.28, slow tempo of 60 BPM, and warm acousticness of 0.92.
```

Two things worth noticing: the hip-hop and jazz songs were retrieved because of the index's related-genre credit (fixing the "exact genre match only" bias from the v1 model card), and Gemini then *judged* among them: it skipped the keyword-strong but vibe-wrong hip-hop track in favor of jazz and ambient, which the rule-based fallback ranked #2 on pure keyword overlap.

**3. `python -m src.main "Drake" -k 3`** (Gemini mode, out-of-catalog artist)

```
Retrieved candidates (top 8):
  - Concrete Kings (hip-hop/confident)  relevance=0.67  matched: confident, electronic, energetic, hip, hop, r&b
  - Velvet Hours (r&b/romantic)         relevance=0.17  matched: r&b
  ...

Recommendations  [source: gemini]
note: No direct matches — interpreted your request as "hip-hop r&b confident moody energetic electronic".

  1. Concrete Kings - Vell Rhymes
     why: Since you are looking for Drake, this hip-hop track matches the genre with a
          confident mood, high energy of 0.78, and a steady tempo of 95 BPM.
  2. Velvet Hours - Mara Soul
     why: This song captures a smooth R&B style similar to Drake's melodic vibe, featuring
          a romantic mood and a moderate tempo of 88 BPM.
```

Drake isn't in the catalog. Instead of failing, Gemini translated the artist into catalog vocabulary, retrieval ran again, and every pick is still a real catalog song. Grounding survives even out-of-catalog requests.

**4. `python -m src.main "high energy workout music" -k 3`** (fallback mode, no API key)

```
Recommendations  [source: fallback]
  note: No GEMINI_API_KEY set — using rule-based fallback.
  1. Storm Runner - Voltline
     why: very close to your energy level; matched your search terms: energy, high, workout
  2. Gym Hero - Max Pulse
     why: very close to your energy level; matched your search terms: energy, high, workout
  3. Neon Warehouse - Pulsewave
     why: very close to your energy level; matched your search terms: energy, high, workout
```

**5. `python -m src.main "sad acoustic songs for a rainy evening" -k 3`** (Gemini mode, real ~7,900-song catalog)

```
Retrieved candidates (top 8):
  - changes (emo rap/sad)      relevance=0.23  matched: sad
  - High Hopes (emo/sad)       relevance=0.23  matched: sad
  - Riot (emo rap/sad)         relevance=0.23  matched: sad
  ...

Recommendations  [source: gemini]
  1. changes - XXXTENTACION
     why: This emo rap track captures a deeply sad mood that fits the reflective
          atmosphere of a rainy evening.
  2. nuts (feat. rainy bear) - Lil Peep
     why: With its sad mood and emo rap genre, this song matches the gloomy vibe
          of a quiet, rainy night.
  3. SAD! - XXXTENTACION
     why: This popular emo rap song delivers the melancholic mood desired for
          listening on a rainy evening.
```

## Reproducible Execution Evidence

Full, unedited terminal logs live in [evidence/](evidence/), each generated by the exact command named in the file. To regenerate any of them, run the command yourself:

| Command | Log | What it proves |
|---|---|---|
| `pytest -q` | [pytest_output.txt](evidence/pytest_output.txt) | 30/30 automated tests pass |
| `python -m src.main "Drake" -k 3` | [interpretation_drake.txt](evidence/interpretation_drake.txt) | Out-of-catalog artist → Gemini interprets the request into catalog vocabulary → grounded picks |
| `python -m src.evaluation` | [evaluation_gemini.txt](evidence/evaluation_gemini.txt) | Golden-query metrics on the live Gemini path (recall@5 = 0.90, 0 hallucinations); runs on the frozen fixture catalog so labels stay valid |
| `python -m src.main "sad acoustic songs for a rainy evening" -k 3` | [real_catalog_query.txt](evidence/real_catalog_query.txt) | Full RAG flow on the real ~7,900-song Spotify catalog |
| `python -m src.main "calm acoustic songs for studying late at night"` | [sample_query_gemini.txt](evidence/sample_query_gemini.txt) | Full RAG flow: retrieval → grounded Gemini picks |
| `python -m src.main "xyzzy quux flurble" -k 3` | [guardrail_nonsense_query.txt](evidence/guardrail_nonsense_query.txt) | Nonsense input degrades gracefully, honest note, no crash |
| `GEMINI_MODEL=nonexistent-model-for-testing python -m src.main "happy upbeat pop songs" -k 3` | [guardrail_api_failure.txt](evidence/guardrail_api_failure.txt) | A failing API call is caught and the rule-based fallback still answers |
| `python -m src.main --like "SAD!" --like "Lucid Dreams" -k 3` | [playlist_similar.txt](evidence/playlist_similar.txt) | Playlist mode: centroid similarity retrieval + grounded Gemini explanations |

The guardrail logs are the interesting ones. Forcing an API failure (bogus model name) produces:

```
Recommendations  [source: fallback]
note: Gemini call failed (ClientError) — used fallback.

1. Sunrise City - Neon Echo
   why: matches your favorite genre (pop); matches your mood (happy); very close to your
        energy level; matched your search terms: happy, pop, upbeat
...
```

And a nonsense query retrieves zero candidates but never crashes:

```
Retrieved candidates (top 0):

Recommendations  [source: fallback]
note: No strong keyword matches in the catalog — showing best rule-based guesses.
```

The third guardrail (the Validator rejecting hallucinated titles and retrying) can't be demoed on demand (Gemini rarely hallucinates against 8 candidates), so it's proven offline by unit tests instead: `test_hallucination_triggers_retry_then_succeeds` and `test_persistent_hallucination_falls_back` in [tests/test_pipeline.py](tests/test_pipeline.py) script a fake Gemini that returns invented titles and assert the retry + fallback behavior.

## Design Decisions

- **RAG over the catalog instead of asking the LLM directly.** An LLM asked "recommend study music" will happily invent songs. Grounding it in retrieved catalog data, and validating the output against that data, makes the answers trustworthy. The retrieval step *actively shapes* the answer: the generator never even sees songs the retriever didn't surface.
- **Hand-rolled TF-IDF instead of embeddings or scikit-learn.** A vector database would hide the mechanics. ~100 lines of TF-IDF + cosine similarity is fully explainable, dependency-free, and fast. Measured on the real catalog: 7,920 songs load and index in 0.2s, and a search takes ~5ms. Trade-off: keyword matching misses paraphrases ("music to fall asleep to" only works because I added derived context tags like "sleep" to low-energy songs).
- **Feature-to-words translation.** Numeric features (energy 0.22) become searchable words ("calm, mellow...") plus activity tags ("sleep, study..."). This is what lets free text match numbers. Trade-off: the vocabulary is hand-curated, so unusual phrasings can still miss.
- **Validator + one corrective retry.** Rather than trusting the prompt, the system checks every returned title against the candidate set. One retry with an explicit correction usually fixes it; persistent failure drops to the fallback rather than showing the user a made-up song.
- **Keeping the v1 scorer as the fallback.** The original recipe still runs when there's no API key, which keeps the app runnable by anyone (including graders) at zero cost, and makes for a nice A/B comparison between templated and generated explanations.
- **Dependency injection in the pipeline.** The generator, validator, retriever, and fallback are constructor arguments, so tests swap in a `FakeGenerator` and run the whole pipeline offline.
- **Ungrounded suggestions are quarantined, not banned.** Users naturally want real-song ideas, but mixing them into the RAG output would silently break grounding. So "Beyond the catalog" is a separate toggleable section (on by default), rendered with an explicit "not validated" disclaimer, so the grounded recommendations never share space with unverified ones.

## Testing Summary

> **Summary: 35/35 automated tests pass. Golden-query recall@5 averaged 0.96 with Gemini (0.90 in fallback mode) with 0 hallucinated titles in both. Real-world testing surfaced six genuine failures along the way, from a retired model ID to Gemini decorating song titles until validation rejected them. Every one of them either improved the system or is now covered by a regression test; the full list is under "What didn't work at first" below.**

The system's reliability is measured three ways: **automated tests** (unit + golden-query metrics), **human evaluation** (the edge-case table below), and **error handling** (every degraded path, whether a missing API key, hallucinated titles, API exceptions, or zero retrieval matches, records *why* it degraded and surfaces that note in the CLI and UI).

**What worked:** 35 unit tests pass, covering retrieval relevance, hallucination detection (including canonicalization of decorated titles), query-to-preference parsing, out-of-catalog query interpretation, playlist similarity, and every pipeline path (valid generation, retry-then-succeed, persistent-hallucination fallback, no-key fallback, API-exception fallback). The golden-query harness (`python -m src.evaluation`) scores 6 realistic queries against human-chosen expected songs:

| Query | Expected found in top 5 | Recall@5 |
|---|---|---|
| calm acoustic songs for studying late at night | 3/4 | 0.75 |
| high energy workout music | 4/4 | 1.00 |
| happy upbeat pop songs | 3/3 (Gemini) · 2/3 (fallback) | 1.00 · 0.67 |
| dark intense metal | 2/2 | 1.00 |
| romantic r&b for a date night | 1/1 | 1.00 |
| sad slow classical music | 1/1 | 1.00 |

**Average recall@5: 0.96 with Gemini, 0.90 in rule-based fallback mode · hallucinated titles: 0 in both.** The golden queries run against the frozen 18-song fixture catalog so the human-chosen labels stay valid regardless of which real dataset ships in `data/songs.csv`. The Gemini path slightly beats the fallback because the LLM recognized Island Time (reggae/uplifting) as a fit for "happy upbeat pop" where exact keyword scoring missed it.

**Human edge-case evaluation** (all inputs actually run against the system; results verified by hand):

| Test input | Evaluation criteria | Result |
|---|---|---|
| "calm acoustic songs for studying late at night" | Top picks are calm, acoustic, study-appropriate | Pass |
| "high energy workout music" | All picks have energy > 0.75 | Pass |
| "romantic r&b for a date night" | Velvet Hours (the only r&b/romantic song) ranked #1 | Pass |
| "music to fall asleep to" (paraphrase) | Retrieves low-energy songs without exact keyword overlap | **Fail at first**: 0 candidates retrieved; fixed by adding "asleep/sleepy/nap" to the sleep tags; now Pass (top 2 = the catalog's two lowest-energy songs) |
| "xyzzy quux flurble" (nonsense) | Degrades gracefully, no crash, honest messaging | Pass: 0 candidates, falls back with a "No strong keyword matches" note; picks are generic |
| "" (empty input) | Handles gracefully, no crash | Pass: fallback over the full catalog (unit-tested); the UI shows a prompt instead of results |
| LLM returns an invented song title (simulated) | Validator rejects it, retries once, then falls back | Pass: unit-tested offline with a fake Gemini |
| LLM returns a real song with a decorated title (live) | Validator accepts the song without accepting hallucinations | **Fail at first**: correct picks like `"Wishing Well" by Juice WRLD` were rejected for their formatting; fixed with title canonicalization; now Pass |
| Playlist of 2 emo rap songs (live) | Similar-song suggestions match the playlist's genre | Pass: all suggestions were emo rap tracks not already in the playlist |

**What didn't work at first:** six honest failures, in the order I hit them.

1. **My own test was wrong.** The initial retrieval test demanded that "Library Rain" rank in the top 4 for the study query. It came 5th, behind four other equally valid study songs. Lesson: when ranking within a cluster of good answers is fuzzy, test that the cluster dominates, not an exact order.
2. **A paraphrase broke retrieval.** "Music to fall asleep to" retrieved zero candidates because the hand-curated vocabulary knew "sleep" but not "asleep". That is keyword retrieval's core weakness. The fix took one line, but only edge-case testing revealed the gap.
3. **The pinned model got retired.** The first live Gemini call failed with a 404 because `gemini-2.5-flash` had been retired for new API keys. The error handling worked as designed (the CLI reported the failure and still returned good picks via fallback), and the permanent fix was the `gemini-flash-latest` alias, which tracks whatever Flash model Google currently ships.
4. **We burned the free-tier quota, and my cache made it worse.** Heavy testing exhausted the daily request quota (429 RESOURCE_EXHAUSTED), which looked like a bug: the sidebar said "Gemini connected" while every search fell back. Worse, the app cached those failed results per query, so one rate-limited call froze that query in fallback mode for the whole session. Two fixes: failed calls are never cached anymore (a `last_transient` flag marks them), and the default model moved to the lite tier, which has a separate, larger quota.
5. **The real dataset had a different shape.** Swapping the 18 fictional songs for a real ~7,900-song Spotify catalog broke assumptions everywhere: different column names, no audio features, multi-genre strings. The loader is now schema-adaptive, missing features load as neutral values that never produce fake search words or fake numbers in Gemini's reasons, and the original 18 songs became a frozen test fixture so the tests and golden labels stay deterministic.
6. **Gemini decorated its answers.** In playlist mode it picked the right songs but returned titles like `"Wishing Well" by Juice WRLD` instead of the bare title, echoing my own prompt formatting, and the Validator rejected every pick. The fix was two-sided: the Validator now canonicalizes decorated titles (a stripped title is only accepted if it is a real candidate, so actual hallucinations still never match), and the prompt explicitly forbids decorating the title field.

**What I learned:** evaluation needs both levels. Unit tests catch broken logic, but only golden queries and live runs told me whether the answers were actually good, and half of these failures (the quota, the retired model, the decorated titles) could never appear in offline tests. "Hallucinated titles: 0" is only meaningful because the harness measures it on every run.

## Reflection

The biggest shift from v1 to v2 was moving from "compute an answer" to "orchestrate and *check* an answer." In v1 the scoring rule *was* the system. In v2 the interesting engineering is around the AI: what data the model is allowed to see (retrieval), how to catch it misbehaving (validation), and how to measure quality over time (golden queries). I also learned that graceful degradation is a design feature, not an afterthought. The fallback path saved the demo three separate times: a missing API key, a retired model, and an exhausted quota. None of those were hypothetical.

> The graded responsible-AI reflection (how I collaborated with AI while building this, one helpful and one flawed AI suggestion, and the system's limitations) is in [model_card.md](model_card.md).
