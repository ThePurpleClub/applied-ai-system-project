# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world platforms like Spotify and YouTube predict what users will love by combining two approaches. Collaborative filtering recommends what similar listeners enjoyed. Content-based filtering recommends songs whose attributes resemble what the user already plays. These platforms learn from signals like plays, skips, saves and replays, then score many candidate songs and re-rank them into a final list that balances relevance with variety. My version is a simple content-based recommender. It doesn't use other users' data. Instead, it scores each song by how closely it matches one user's taste profile. It gives the most weight to **genre**, then **mood** and **energy**, with **valence** as a tiebreaker. I left out tempo, danceability and acousticness because in this dataset they are highly correlated with energy and would mostly count the same signal more than once. A separate ranking step then sorts the scored songs and picks the top results, aiming for recommendations that are accurate, easy to explain and not repetitive.

### Song Features and User Profile

- **`Song` features used in scoring:** `genre`, `mood`, `energy`, `valence`. The other columns (`tempo_bpm`, `danceability`, `acousticness`, `instrumentalness`, `popularity`) are loaded but not scored.
- **`UserProfile` stores:** `favorite_genre`, `favorite_mood`, `target_energy` and `target_valence` (default 0.65, the dataset average, meaning no preference). It also has `likes_acoustic`, `prefers_instrumental` and `prefers_popular`, which are reserved for later experiments and not used in the score yet.

### Algorithm Recipe

Each song is scored against the user profile, then songs are sorted by score (highest first) and the top `k` are returned.

| Rule | Points | Notes |
|---|---|---|
| Genre match | **+2.0** | Exact match, case-insensitive |
| Related genre | **+1.0** | Instead of the full match, when one genre contains the other (e.g. `pop` and `indie pop`) |
| Mood match | **+1.0** | Exact match, case-insensitive |
| Energy similarity | **0 to +1.5** | `1.5 × (1 − |song.energy − target_energy|)` |
| Valence similarity | **0 to +0.5** | `0.5 × (1 − |song.valence − target_valence|)`, used as a tiebreaker |

Maximum score is **5.0**. Every rule that awards points adds a short reason (e.g. "genre match (+2.0)") so each recommendation can be explained.

**Why these weights**

- Genre counts double compared to mood because genre is a more stable preference, and mood is already partly captured by energy and valence (in this dataset `chill` songs sit around 0.3–0.4 energy, `intense` songs around 0.9).
- Energy is capped at 1.5 so that a song with a close energy level can never outscore an exact genre match on its own.
- Most songs in the catalog match neither the genre nor the mood (20 songs, about 17 genres and 15 moods), so energy and valence decide the ranking among them.

**Expected Biases**

- **Genre over-prioritized:** this system might over-prioritize genre and overlook great songs that match the user's mood. A pop song with the wrong mood (e.g. *Gym Hero*, `intense`, for a happy pop fan) can outrank a song with the right mood in a related genre. The related-genre rule reduces this but doesn't remove it.
- **Exact label matching:** genre and mood only count when the text matches. `sad` and `melancholic` count as different moods, and `r&b` gets no credit for a `soul` fan, so songs with similar feel but different labels lose points.
- **Small, uneven catalog:** `lofi` has 3 songs while most genres have 1. Lofi fans get a full list of strong matches; fans of a single-song genre like `metal` get one match and then songs ranked mostly by energy.
- **Filter bubble:** songs in the user's genre always come first, so the system rarely suggests anything new. Nothing in the score rewards variety.
- **Mid-energy songs favored:** for a user near the middle of the energy range, no song can be more than about 0.5 away, so energy barely separates songs. Users at the extremes (very calm or very intense) see stronger energy effects.

### Data Flow

```mermaid
flowchart TD
    A["<b>Input</b><br/>user_prefs<br/>genre, mood, energy, valence"] --> C
    B["data/songs.csv"] -->|"load_songs()"| L["list of 20 song dicts"]
    L --> C

    subgraph P["<b>Process</b>: the loop in recommend_songs()"]
        C["for each song"] --> S["score_song(user_prefs, song)<br/>genre +2.0 / +1.0<br/>mood +1.0<br/>energy 0 to 1.5<br/>valence 0 to 0.5"]
        S --> R["(song, score, reasons)"]
        R -->|next song| C
    end

    R -->|all songs scored| O["<b>Output</b><br/>sort by score, highest first<br/>keep top k"]
    O --> M["main.py prints<br/>title, score, explanation"]
```

1. **Input:** `main.py` builds `user_prefs`, and `load_songs()` reads the CSV into a list of dicts, converting numeric columns to `float`/`int`.
2. **Process:** `recommend_songs()` loops over every song and calls `score_song()`, which returns `(score, reasons)`. Each song is judged on its own and no song is skipped, so the loop yields one `(song, score, explanation)` per song.
3. **Output:** the scored list is sorted by score (highest first) and cut to the top `k`. `main.py` prints each song with its score and the joined reasons.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Output of `python -m src.main` for the default profile in `src/main.py`, a late-night studier who wants calm lofi music (genre `lofi`, mood `chill`, energy 0.38, valence 0.60):

```
$ python -m src.main
Loaded 20 songs from data/songs.csv

============================================================
  Top Recommendations
  Profile: lofi / chill / energy 0.38
============================================================

1. Library Rain by Paper Lanterns
   Score: 4.96 / 5.00   (lofi, chill)
   Why:
     - genre match (+2.0)
     - mood match (+1.0)
     - energy 0.35 vs target 0.38 (+1.46)
     - valence 0.60 vs target 0.60 (+0.50)

2. Midnight Coding by LoRoom
   Score: 4.92 / 5.00   (lofi, chill)
   Why:
     - genre match (+2.0)
     - mood match (+1.0)
     - energy 0.42 vs target 0.38 (+1.44)
     - valence 0.56 vs target 0.60 (+0.48)

3. Focus Flow by LoRoom
   Score: 3.96 / 5.00   (lofi, focused)
   Why:
     - genre match (+2.0)
     - energy 0.40 vs target 0.38 (+1.47)
     - valence 0.59 vs target 0.60 (+0.49)

4. Spacewalk Thoughts by Orbit Bloom
   Score: 2.83 / 5.00   (ambient, chill)
   Why:
     - mood match (+1.0)
     - energy 0.28 vs target 0.38 (+1.35)
     - valence 0.65 vs target 0.60 (+0.47)

5. Coffee Shop Stories by Slow Stereo
   Score: 1.93 / 5.00   (jazz, relaxed)
   Why:
     - energy 0.37 vs target 0.38 (+1.48)
     - valence 0.71 vs target 0.60 (+0.45)

============================================================
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



