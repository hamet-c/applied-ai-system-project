# 🎧 Model Card: Music Recommender Simulation

> Sections 1–9 describe the original rule-based **VibeFinder 1.0** (Modules 1–3), which still lives inside the project as the fallback scorer. Sections 10–13 are the responsible-AI reflection for the RAG-powered **Songly** (see README).

## 1. Model Name  

**VibeFinder 1.0**

A tiny music recommender that matches songs to your vibe.

---

## 2. Intended Use  

VibeFinder suggests songs from a small catalog based on a user's taste.

You give it a favorite genre, a mood, and how much energy you want. It hands back the top 5 songs that fit, plus a short reason for each one. It assumes the user can describe their taste in those simple terms.

This is a classroom project for learning how recommenders work. It is not built for real users or a real music app.

---

## 3. How the Model Works  

Think of it like a judge giving each song points.

A song earns 2 points if its genre matches what you asked for. It earns 1 point if the mood matches. It earns up to 1 more point for having energy close to what you want, so a perfect energy match is worth a full point and a far-off one is worth almost nothing. There is also a small half-point for matching whether you like acoustic music.

Every song gets scored this way. Then the list is sorted from highest score to lowest, and the top 5 come back. The main change I made from the starter code was writing the actual scoring rule and turning the weights into settings I could easily change for experiments.

---

## 4. Data  

The catalog has 18 songs. I started with 10 and used AI to help me generate 8 more so there was more variety to test with.

Each song has a genre, a mood, and number features from 0 to 1 for energy, valence, danceability, and acousticness, plus a tempo. There are around 15 genres (pop, lofi, rock, jazz, metal, reggae, and more) and many moods.

The dataset is very small, so most genres only have one or two songs. It also has nothing about lyrics, language, artist popularity, or how songs actually sound to a person, so a lot of real musical taste is missing.

---

## 5. Strengths  

The system works best when a user's taste lines up with songs that exist in the catalog.

For clear profiles like "chill lofi" or "high-energy pop," the top picks feel right. The genre, mood, and energy all point the same way and the obvious song wins. The energy score also does a good job separating loud songs from calm ones, so opposite profiles get genuinely different lists. And every recommendation comes with a reason, which makes it easy to see why a song showed up.

---

## 6. Limitations and Bias 

The biggest weakness I found is that the system leans too hard on exact genre and mood matches. Because genre is worth 2.0 points and mood 1.0, but energy can only ever add up to 1.0, a song that shares your genre almost always beats a song that only matches your energy or mood, even when the energy match is much better. This creates a filter bubble: once a genre matches, the recommender stops caring how well the rest of the song fits you. It also punishes users whose taste crosses genre lines, since a "rock" fan gets zero credit for a metal or punk track that a real listener would probably enjoy. Underrepresented genres suffer too. When a user picks a genre with only one matching song (like metal), the entire ranking below that one song is decided by the energy number alone, which is a much weaker signal. Finally, the scoring only reads exact string matches, so a mood like "sad" that does not exist in the catalog silently contributes nothing rather than being handled.

---

## 7. Evaluation  

### Profiles tested

I tested five profiles: three normal personas and two adversarial edge cases designed to try to trick the scorer.

- **High-Energy Pop** (`genre=pop, mood=happy, energy=0.9`)
- **Chill Lofi** (`genre=lofi, mood=chill, energy=0.35`)
- **Deep Intense Rock** (`genre=rock, mood=intense, energy=0.9`)
- **Conflicting (loud + sad)** (`genre=pop, mood=sad, energy=0.95`) - an energy/mood combo that rarely exists in real music
- **Impossible (metal + relaxed)** (`genre=metal, mood=relaxed, energy=0.5`) - a genre/mood pairing with zero matching songs in the catalog

### Output

```
====================================================
  High-Energy Pop   (genre=pop mood=happy energy=0.9)
====================================================
  1. Sunrise City - Neon Echo        score: 3.92  (genre; mood; energy)
  2. Gym Hero - Max Pulse            score: 2.97  (genre; energy)
  3. Rooftop Lights - Indigo Parade  score: 1.86  (mood; energy)
  4. Storm Runner - Voltline         score: 0.99  (energy)
  5. Neon Warehouse - Pulsewave      score: 0.94  (energy)

====================================================
  Chill Lofi   (genre=lofi mood=chill energy=0.35)
====================================================
  1. Library Rain - Paper Lanterns   score: 4.00  (genre; mood; energy)
  2. Midnight Coding - LoRoom         score: 3.93  (genre; mood; energy)
  3. Focus Flow - LoRoom              score: 2.95  (genre; energy)
  4. Spacewalk Thoughts - Orbit Bloom score: 1.93  (mood; energy)
  5. Coffee Shop Stories - Slow Stereo score: 0.98 (energy)

====================================================
  Deep Intense Rock   (genre=rock mood=intense energy=0.9)
====================================================
  1. Storm Runner - Voltline         score: 3.99  (genre; mood; energy)
  2. Gym Hero - Max Pulse            score: 1.97  (mood; energy)
  3. Neon Warehouse - Pulsewave      score: 0.94  (energy)
  4. Iron Verdict - Blacklight Forge score: 0.92  (energy)
  5. Sunrise City - Neon Echo        score: 0.92  (energy)

====================================================
  Conflicting (loud + sad)   (genre=pop mood=sad energy=0.95)
====================================================
  1. Gym Hero - Max Pulse            score: 2.98  (genre; energy)
  2. Sunrise City - Neon Echo        score: 2.87  (genre; energy)
  3. Neon Warehouse - Pulsewave      score: 0.99  (energy)
  4. Iron Verdict - Blacklight Forge score: 0.97  (energy)
  5. Storm Runner - Voltline         score: 0.96  (energy)

====================================================
  Impossible (metal + relaxed)   (genre=metal mood=relaxed energy=0.5)
====================================================
  1. Iron Verdict - Blacklight Forge score: 2.52  (genre)
  2. Coffee Shop Stories - Slow Stereo score: 1.87 (mood; energy)
  3. Paper Moon Dreams - Wisp        score: 1.00  (energy)
  4. Velvet Hours - Mara Soul        score: 0.95  (energy)
  5. Dust and Pine - Old Creek Road  score: 0.94  (energy)
```

### Comparing profiles

- **High-Energy Pop vs Chill Lofi:** These are near opposites and the output confirms the scorer is actually reading the profile, not just returning the same list. Pop pulls loud, upbeat tracks (Sunrise City, Gym Hero) to the top; Lofi pulls slow, mellow tracks (Library Rain, Midnight Coding). Nothing overlaps in the top two. This is the clearest evidence the energy term is doing real work, since both top picks also match on genre and mood.
- **High-Energy Pop vs Deep Intense Rock:** Same high energy (0.9) but different genre/mood. Gym Hero appears high in both because it is loud and intense, but each list is led by its own genre match (Sunrise City for pop, Storm Runner for rock). This shows genre is the deciding factor when energy is tied, which matches the intended recipe.
- **Deep Intense Rock vs Conflicting (loud + sad):** Both want high energy, but the rock profile has a real mood match and the conflicting one does not. The conflicting profile's top songs (Gym Hero, Sunrise City) win purely on genre plus energy, and "sad" adds nothing because no song is tagged sad. This makes sense and correctly shows the scorer degrades gracefully rather than crashing on a mood that is missing from the data.
- **Deep Intense Rock vs Impossible (metal + relaxed):** The rock profile finds a perfect three-way match (Storm Runner). The impossible profile cannot, so its #1 is a metal song that matches genre only, while #2 is a jazz song that matches the relaxed mood and energy. This split top result is the most interesting outcome, because the scorer is torn between two half-matches and neither feels clearly right.

### What surprised me

The "Impossible" profile was the most revealing. A loud metal track (Iron Verdict) ranked #1 for someone asking for *relaxed* music, purely because genre outweighs everything else. That does not match musical intuition at all, and it exposed the genre-over-everything bias described in section 6.

### Data experiment

I doubled the energy weight (1.0 to 2.0) and halved the genre weight (2.0 to 1.0), then re-ran all profiles. For the three normal profiles the top pick did not change, because their #1 song matched genre, mood, and energy all at once. But the "Impossible (metal + relaxed)" profile flipped: its #1 changed from Iron Verdict (metal, loud) to Coffee Shop Stories (relaxed jazz), because energy closeness now outweighed the lone genre match. The change made that specific result feel more accurate, since a relaxed listener probably does want the calm song. It confirmed the system is highly sensitive to the genre weight and that lowering it reduces the filter-bubble effect.

---

## 8. Future Work  

A few things I would change if I kept going:

- Let a genre match count partly for related genres, so a rock fan gets some credit for metal or punk.
- Use more of the number features I already have, like danceability and valence, instead of just energy.
- Add diversity to the top 5 so the same artist does not show up twice, and grow the catalog so rare genres have real competition.

---

## 9. Personal Reflection  

My biggest learning moment was seeing that a recommender is really just a scoring rule plus a sort. Once I split "score one song" from "rank all the songs," the whole thing clicked and felt much less mysterious.

AI tools helped me move fast. I used them to generate extra songs for the catalog, to talk through how to weight genre against mood, and to write the CSV loading. But I still had to double-check the results, like when a loud metal song ranked first for a "relaxed" user. The AI would happily write working code, but deciding whether the output actually made sense was on me.

What surprised me most was how something this simple still feels like a real recommendation. There is no machine learning here, just points and a sort, but the reasons it gives make it feel smart. If I extended it, I would fix the genre bias first, since that was the clearest flaw I found.

---

## 10. Limitations and Biases (Songly)

The retrieval is keyword-based, so it only understands words I thought to put in the index. "Music to fall asleep to" returned *nothing* until I added "asleep" to the vocabulary. Embeddings would handle paraphrases like that; keywords never fully will. The catalog is now a real ~7,900-song Spotify dataset, but it has no audio features (energy, valence, tempo), so vibe matching leans entirely on genre strings and moods derived from genre keywords. A "mellow acoustic" query can only match songs whose *genres* say something like that. There is also a data-quality problem I only found by measuring the catalog directly: 3,105 of the 7,920 songs (39%) carry no genre tag at all, and 1,578 songs (20%) end up with four or fewer searchable words in their index document, barely more than their title and artist. Those songs are effectively unreachable unless someone searches their exact title, so a fifth of the catalog is invisible to normal use. Results still look good because the well-tagged songs dominate, which is exactly what makes this kind of gap easy to miss. The catalog also skews toward popular Western music, and the popularity boost in ranking reinforces that bias. The playlist feature inherits it too: similar-song suggestions come from the same index, so a niche playlist gets pulled toward whatever the catalog has the most of. The system is English-only. And the LLM explanations are a subtle risk: they sound equally confident whether the match is great or mediocre, because Gemini is told to justify whatever the retriever handed it. It never says "honestly, nothing here fits."

## 11. Potential Misuse and Prevention

Two realistic ones. First, a recommender like this launders whatever is in its data: if the catalog were curated to push certain artists (pay-for-play), the AI would generate confident, factual-sounding reasons for rigged picks. The "What was retrieved" panel is my main defense, since users can see exactly what the model was given and judge the grounding themselves. Second, prompt injection: song metadata gets pasted into the LLM prompt, so a malicious "song title" could try to steer the model. The Validator limits the damage, because output titles must exist in the retrieved set and the strict JSON format rejects anything else. Keeping the source badge visible (gemini vs fallback) also means nobody mistakes templated output for AI judgment or vice versa.

There's also a boundary I had to defend *against myself*: I wanted the app to recommend real songs too, but mixing unverified LLM knowledge into the grounded results would quietly turn the RAG system back into a plain chatbot. The compromise is a "Beyond the catalog" section (on by default, toggleable off) that is visually separate and explicitly labeled as not retrieved and not validated. The user can always tell which recommendations the system can actually vouch for.

## 12. What Surprised Me While Testing Reliability

Four things genuinely surprised me. First, my initial live Gemini call failed with a 404 because the model I'd pinned (`gemini-2.5-flash`) had already been retired for new API keys. The fallback caught it and the app kept working, which honestly proved the architecture better than any test I wrote. Second, how many failures had nothing to do with code being wrong: we exhausted the free-tier daily quota during testing, and the app looked broken ("Gemini connected" in the sidebar, fallback on every search) when the real problem was rate limits plus my cache freezing failed results. Reliability turned out to be as much about quotas, retired models, and caching policy as about logic. Third, recall@5 barely moved between Gemini mode (0.96) and fallback mode (0.90). Retrieval decides *what's findable*; the LLM mostly decides ordering and explanations. Its real value showed up in judgment, like skipping a keyword-strong hip-hop track for jazz and ambient on a "romantic date night" query. Fourth, the strangest failure: in playlist mode Gemini chose the right songs but formatted their titles wrong (`"Wishing Well" by Juice WRLD`), copying the style of my own prompt, and the Validator rejected every pick. The model can be right and unusable at the same time, and the guardrail has to tell those apart.

## 13. My Collaboration with AI

I built this project working with an AI coding assistant (Claude), giving it the assignment requirements step by step and reviewing what it produced.

**One helpful suggestion:** translating the numeric features into searchable words (energy 0.22 becomes "calm, mellow") plus related-genre credit in the index (a metal query also surfaces rock). That one design idea fixed the exact genre-bias flaw documented in section 6 of this card, and it made free-text search possible without any embedding model. It also survived the dataset swap: when the real catalog turned out to have no audio features at all, the same idea ran in reverse, deriving moods from genre keywords instead.

**One flawed suggestion:** the AI wrote the generator with `gemini-2.5-flash` hardcoded as the default model, which was already deprecated for new API keys, so the first real API call 404'd. It looked completely reasonable in code review; only running it live exposed the problem. It wasn't the only flaw I caught, either: an early version of the app cached results even when the Gemini call had failed, which froze one rate-limited query in fallback mode for a whole session, and an early test asserted an exact ranking position that failed for a song ranked 5th among five equally valid answers. Every fix was easy, but every one required *me* to run the system and judge the output. The AI's code always looked right, and looking right isn't the same as being right.
