# 🎵 VibeFinder 2.0 — a RAG-Powered Music Recommender

## Original Project (Modules 1–3)

This project evolved from the **Music Recommender Simulation ("VibeFinder 1.0")**. The original version represented songs and a user taste profile as data, scored every song in an 18-track catalog with a hand-tuned weighted recipe (genre +2.0, mood +1.0, energy closeness up to +1.0, acousticness +0.5), and printed the top 5 picks with templated explanations. It only accepted rigid, pre-defined profiles (`genre=pop, mood=happy, energy=0.9`) and its evaluation exposed a clear bias: exact genre matches outweighed everything else, so a loud metal track ranked #1 for a user asking for *relaxed* music.

## Title and Summary

**VibeFinder 2.0** turns that rule-based scorer into a **Retrieval-Augmented Generation (RAG)** system with a Streamlit web UI. You describe what you want to hear in your own words — *"calm acoustic songs for studying late at night"* — and the system:

1. **Retrieves** the most relevant songs from the catalog using a TF-IDF search index,
2. **Generates** ranked picks with conversational explanations using the Gemini API, grounded in **only** the retrieved songs,
3. **Validates** the output, rejecting any hallucinated song title and retrying, and
4. **Falls back** to the original rule-based scorer when no API key is set or generation fails.

Why it matters: this is the same architecture real AI products use to keep LLMs honest. The model never answers from its imagination — it answers from retrieved data, and a validator plus an evaluation harness measure that it stays that way.

## Architecture Overview

The full UML diagram is in [diagrams/architecture.mmd](diagrams/architecture.mmd). In short:

```
User query → Retriever (TF-IDF over SongIndex) → Candidates
           → GeminiGenerator (grounded prompt: candidates ONLY)
           → Validator (hallucinated title? → retry once → fallback)
           → Recommendation[] → Streamlit UI
```

- **`SongCatalog` / `SongIndex`** ([src/catalog.py](src/catalog.py), [src/index.py](src/index.py)) — each song becomes a searchable text document: title, artist, genre, mood, plus its numeric features translated into words (energy 0.9 → "energetic, high-energy...") and derived activity tags (low energy + acoustic → "study, focus..."). Important fields are repeated to act as a boost, and related genres get partial credit (a metal query also surfaces rock).
- **`Retriever`** ([src/retriever.py](src/retriever.py)) — cosine similarity between the query and every song document; returns the top-8 candidates with their matched terms.
- **`GeminiGenerator`** ([src/generator.py](src/generator.py)) — builds a grounded prompt containing *only* the candidates and asks Gemini for strict-JSON picks with one-sentence reasons.
- **`Validator`** ([src/validator.py](src/validator.py)) — checks every returned title against the candidate set; hallucinations trigger one corrective retry, then fallback.
- **`FallbackScorer`** ([src/fallback.py](src/fallback.py)) — the original v1 scoring recipe, now driven by a taste profile derived from the query text. Runs when there's no API key or generation fails.
- **`RecommenderPipeline`** ([src/pipeline.py](src/pipeline.py)) — orchestrates the flow above; the CLI, UI, tests, and evaluator all share it.
- **`EvaluationHarness`** ([src/evaluation.py](src/evaluation.py)) — golden-query testing (see Testing Summary).

Three checkpoints watch the AI's output: the **Validator** at runtime, the **EvaluationHarness** before changes ship, and the **human** — the UI shows a "What was retrieved" panel so you can compare the picks against the retrieval set and judge the grounding yourself.

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

3. *(Optional but recommended)* Enable Gemini generation — copy `.env.example` to `.env` and paste in a free API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey):

   ```
   GEMINI_API_KEY=your-key-here
   ```

   Without a key the app still works fully, using the rule-based fallback.

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
   pytest                      # 25 unit tests, all offline
   python -m src.evaluation    # golden-query metrics
   ```

## Sample Interactions

Real CLI output, captured in fallback mode (no API key). With a key set, the picks come from Gemini with generated one-sentence reasons instead of templated ones, and the source badge reads `gemini`.

**1. `python -m src.main "calm acoustic songs for studying late at night"`**

```
Retrieved candidates (top 8):
  - Spacewalk Thoughts (ambient/chill)   relevance=0.26  matched: acoustic, calm, late, night, studying
  - Autumn Nocturne (classical/melancholic)  relevance=0.24  matched: acoustic, calm, late, night, studying
  - Midnight Coding (lofi/chill)         relevance=0.21  matched: acoustic, late, night, studying
  - Focus Flow (lofi/focused)            relevance=0.21  matched: acoustic, late, night, studying
  - Library Rain (lofi/chill)            relevance=0.21  matched: acoustic, late, night, studying
  ...

Recommendations  [source: fallback]
  1. Focus Flow - LoRoom
     why: matches your mood (focused); very close to your energy level;
          matches your acoustic preference; matched your search terms: acoustic, late, night, studying
  2. Spacewalk Thoughts - Orbit Bloom
     why: very close to your energy level; matches your acoustic preference;
          matched your search terms: acoustic, calm, late, night, studying
  3. Autumn Nocturne - The Hallberg Quartet
     ...
```

**2. `python -m src.main "high energy workout music" -k 3`**

```
Recommendations  [source: fallback]
  1. Storm Runner - Voltline
     why: very close to your energy level; matched your search terms: energy, high, workout
  2. Gym Hero - Max Pulse
     why: very close to your energy level; matched your search terms: energy, high, workout
  3. Neon Warehouse - Pulsewave
     why: very close to your energy level; matched your search terms: energy, high, workout
```

**3. `python -m src.main "romantic r&b for a date night" -k 3`**

```
Retrieved candidates (top 8):
  - Velvet Hours (r&b/romantic)        relevance=0.73  matched: r&b, romantic
  - Concrete Kings (hip-hop/confident) relevance=0.10  matched: night, r&b
  - Coffee Shop Stories (jazz/relaxed) relevance=0.10  matched: night, r&b
  ...

Recommendations  [source: fallback]
  1. Velvet Hours - Mara Soul
     why: matches your favorite genre (r&b); matches your mood (romantic); matched your search terms: r&b, romantic
  2. Concrete Kings - Vell Rhymes
     why: matched your search terms: night, r&b
```

Note in example 3 how the hip-hop and jazz songs matched "r&b" — that's the related-genre credit in the index, which directly fixes the "exact genre match only" bias documented in the v1 model card.

## Design Decisions

- **RAG over the catalog instead of asking the LLM directly.** An LLM asked "recommend study music" will happily invent songs. Grounding it in retrieved catalog data — and validating the output against that data — makes the answers trustworthy. The retrieval step *actively shapes* the answer: the generator never even sees songs the retriever didn't surface.
- **Hand-rolled TF-IDF instead of embeddings or scikit-learn.** With an 18-song catalog, a vector database would be overkill and would hide the mechanics. ~100 lines of TF-IDF + cosine similarity is fully explainable, dependency-free, and fast. Trade-off: keyword matching misses paraphrases ("music to fall asleep to" only works because I added derived context tags like "sleep" to low-energy songs).
- **Feature-to-words translation.** Numeric features (energy 0.22) become searchable words ("calm, mellow...") plus activity tags ("sleep, study..."). This is what lets free text match numbers. Trade-off: the vocabulary is hand-curated, so unusual phrasings can still miss.
- **Validator + one corrective retry.** Rather than trusting the prompt, the system checks every returned title against the candidate set. One retry with an explicit correction usually fixes it; persistent failure drops to the fallback rather than showing the user a made-up song.
- **Keeping the v1 scorer as the fallback.** The original recipe still runs when there's no API key, which keeps the app runnable by anyone (including graders) at zero cost, and makes for a nice A/B comparison between templated and generated explanations.
- **Dependency injection in the pipeline.** The generator, validator, retriever, and fallback are constructor arguments, so tests swap in a `FakeGenerator` and run the whole pipeline offline.

## Testing Summary

> **Summary: 27/27 automated tests pass. Golden-query recall@5 averaged 0.90 with 0 hallucinated titles. Edge-case evaluation found one real failure — a paraphrased query ("music to fall asleep to") retrieved nothing because the index vocabulary lacked the word "asleep". Accuracy improved after expanding the derived tags; the case is now a regression test.**

The system's reliability is measured three ways: **automated tests** (unit + golden-query metrics), **human evaluation** (the edge-case table below), and **error handling** (every degraded path — missing API key, hallucinated titles, API exceptions, zero retrieval matches — records *why* it degraded and surfaces that note in the CLI and UI).

**What worked:** 27 unit tests pass, covering retrieval relevance, hallucination detection, query-to-preference parsing, and every pipeline path (valid generation, retry-then-succeed, persistent-hallucination fallback, no-key fallback, API-exception fallback). The golden-query harness (`python -m src.evaluation`) scores 6 realistic queries against human-chosen expected songs:

| Query | Expected found in top 5 | Recall@5 |
|---|---|---|
| calm acoustic songs for studying late at night | 3/4 | 0.75 |
| high energy workout music | 4/4 | 1.00 |
| happy upbeat pop songs | 2/3 | 0.67 |
| dark intense metal | 2/2 | 1.00 |
| romantic r&b for a date night | 1/1 | 1.00 |
| sad slow classical music | 1/1 | 1.00 |

**Average recall@5: 0.90 · hallucinated titles: 0** (fallback mode).

**Human edge-case evaluation** (all inputs actually run against the system; results verified by hand):

| Test input | Evaluation criteria | Result |
|---|---|---|
| "calm acoustic songs for studying late at night" | Top picks are calm, acoustic, study-appropriate | Pass |
| "high energy workout music" | All picks have energy > 0.75 | Pass |
| "romantic r&b for a date night" | Velvet Hours (the only r&b/romantic song) ranked #1 | Pass |
| "music to fall asleep to" (paraphrase) | Retrieves low-energy songs without exact keyword overlap | **Fail at first** — 0 candidates retrieved; fixed by adding "asleep/sleepy/nap" to the sleep tags; now Pass (top 2 = the catalog's two lowest-energy songs) |
| "xyzzy quux flurble" (nonsense) | Degrades gracefully, no crash, honest messaging | Pass — 0 candidates, falls back with a "No strong keyword matches" note; picks are generic |
| "" (empty input) | Handles gracefully, no crash | Pass — fallback over the full catalog (unit-tested); the UI shows a prompt instead of results |
| LLM returns an invented song title (simulated) | Validator rejects it, retries once, then falls back | Pass — unit-tested offline with a fake Gemini |

**What didn't work at first:** two honest failures. (1) My initial retrieval test demanded that "Library Rain" rank in the top 4 for the study query — it came 5th, behind four other *equally valid* study songs. The lesson: when ranking within a cluster of good answers is fuzzy, test that the cluster dominates, not an exact order. (2) The paraphrased query "music to fall asleep to" retrieved zero candidates, because the hand-curated index vocabulary knew "sleep" but not "asleep" — keyword retrieval's core weakness. The fix (expanding the derived tags) took one line, but only edge-case testing revealed the gap.

**What I learned:** evaluation needs *both* levels — unit tests catch broken logic, but only golden queries told me whether the system's answers were actually good. And "hallucinated titles: 0" is only meaningful because the harness measures it on every run.

## Reflection

The biggest shift from v1 to v2 was moving from "compute an answer" to "orchestrate and *check* an answer." In v1 the scoring rule *was* the system. In v2 the interesting engineering is around the AI: what data the model is allowed to see (retrieval), how to catch it misbehaving (validation), and how to measure quality over time (golden queries). I also learned that graceful degradation is a design feature, not an afterthought — the fallback path means the demo never dies on a missing API key.

> The graded responsible-AI reflection — how I collaborated with AI while building this, one helpful and one flawed AI suggestion, and the system's limitations — is in [model_card.md](model_card.md).
