# AI Interactions Log

> **Stretch features only.** Only fill in the sections that apply to stretch features you attempted. If you did not attempt a stretch feature, leave its section blank or delete it. This file is not required for the core project.

---

## Agentic Workflow (SF8)

> Document your experience using an AI agent (e.g., Cursor Agent, Claude, Copilot) to make multi-step changes autonomously.

**What task did you give the agent?**

I used Claude Code (an AI coding agent in VS Code) to add 5 new song attributes to the dataset and make the scoring use them. This one task touched the data file, the scoring code, the CLI runner and this log. Earlier in the project I also used the agent to apply a change to the scoring weights and check that the math still worked (prompt 2).

**Prompts used:**

1. The main task prompt:
   > Introduce 5 or more complex attributes to your dataset that are not currently present in the baseline data, such as Song Popularity (0-100), Release Decade, or Detailed Mood Tags (e.g., "nostalgic," "aggressive," "euphoric"). Update both data/songs.csv and the scoring logic in src/recommender.py so scoring accounts for the new attributes.

2. An earlier prompt the agent gave a *sub-agent* to apply a weight change. I'm including it because it shows how to ask for verification, not just code:
   > Change the weights in `score_song` to a "middle ground": exact genre +1.5, related genre +0.75, mood +1.0 (unchanged), energy `2.0 * (1 - abs(song_energy - target_energy))`, valence unchanged. The number in each reason string must equal the points actually added. […] Verify the math is still valid: (1) a song identical to a profile scores exactly 5.0; (2) for every song × every profile, 0.0 <= score <= 5.0, including the energy 1.5 profile; (3) the numbers in the reason strings sum to the returned score; (4) hand-computed spot check: Sad Banger vs Last Train Blues should be 3.975; (5) pytest passes and `python -m src.main` runs.

**What did the agent generate or change?**

*`data/songs.csv`:* added 5 new columns to all 20 songs. The values are hand-assigned, since the songs are fictional.

| Attribute | Example values | Notes |
|---|---|---|
| `release_decade` | 1950 … 2020 | 8 different decades |
| `mood_tags` | `nostalgic\|late-night\|brooding` | 3 detailed tags per song, separated by `\|` |
| `language` | `en`, `es`, `ko`, `none` | `none` = instrumental |
| `explicit` | `true` / `false` | 3 songs are explicit (Gym Hero, Iron Cathedral, Concrete Sonnets) |
| `lyrical_theme` | `heartbreak`, `celebration`, `rebellion`, `none` | `none` = instrumental |

*`src/recommender.py`:*
- `load_songs()` now converts `release_decade` to a number and `explicit` to true/false.
- The `Song` dataclass has the 5 new fields, with defaults so the existing tests still work.
- `score_song()` has 5 new rules. **Each one only scores if the listener actually sets that preference.** I asked for this on purpose: in my bias analysis, the default valence of 0.65 quietly penalized sad songs for users who never mentioned valence, and I didn't want the new attributes to repeat that.

  | Listener preference | Points |
  |---|---|
  | `mood_tags: [...]` | +0.25 per shared tag, up to +0.75 |
  | `decade: 1980` | +0.5 same decade, +0.25 next-door decade |
  | `language: "en"` | +0.25 if it matches |
  | `themes: [...]` | +0.25 if the song's theme is in the list |
  | `allow_explicit: False` | −2.0 for explicit songs (a penalty, not a filter) |
- New `max_score(user_prefs)` function: the best possible score is 5.0 plus whatever optional preferences the listener set, so it's no longer always 5.

*`src/main.py`:*
- Added 3 profiles that use the new preferences: *Retro Night Drive*, *Melancholy Tags* and *Family Road Trip*.
- Changed the printed `/ 5.00` to each profile's real maximum.

**What did you verify or fix manually?**

- **Old results didn't change.** I ran the last committed version and the new version, and the output for the original 9 profiles is *byte-for-byte identical*. This confirms the new rules really are optional and don't change anything for listeners who don't use them.
- **The math adds up.**
  - For all 240 song × profile pairs, the points listed in the "Why" lines add up to the score (within 0.005 rounding).
  - No score is above that profile's `max_score`.
  - A made-up "perfect" song scores exactly `max_score` for all 12 profiles.
  - Hand check: Night Drive Loop for *Retro Night Drive* = 1.5 + 1.0 + 1.8 + 0.48 + 0.5 (two tags) + 0.25 (language) = **5.53**, and the code gives 5.53.
- **Data types are right.** `explicit` loads as true/false, not as the text `"false"`. That matters because in Python any non-empty text, even `"false"`, counts as true, so every song would have been treated as explicit.
- **`pytest` still passes** (2 tests).
- **The results make sense.**
  - *Melancholy Tags:* Nocturne in Grey (classical, mood "melancholic") jumps to #2 because its tags include "melancholic". Before, it got no mood credit for a "sad" listener. The tags fix the exact-label problem I found earlier.
  - *Retro Night Drive:* Night Drive Loop scores 5.53. Gravel Road Gospel comes 2nd thanks to its "nostalgic" tag and a 1970s release date, which is next door to the 1980s.
- **Things I noticed and decided to keep:**
  - *Explicit is a penalty, not a filter.* Gym Hero (explicit) still shows up at #5 for *Family Road Trip*, because there are only 20 songs and the penalty only drops it from 4.32 to 2.32. A real "kids in the car" setting should probably *remove* explicit songs instead.
  - *Scores can now be slightly negative.* The lowest is −0.03, and it only comes from the explicit penalty. Earlier I had said scores can't go negative; that's now only true when `allow_explicit` isn't set.
  - *No commas in the "Why" lines.* `main.py` splits each song's reasons on ", ", so the new reasons use "/" to join tags (e.g. `mood tags nostalgic/late-night`). Using commas would have broken the printout.
  - *The attribute values are my judgment calls.* Which decade or tags a fictional song gets is subjective, so recommendations based on them are only as good as those labels.
- **An earlier mistake by the agent.** During the weight experiments, one of its automated find-and-replace edits accidentally changed the mood weight along with the genre weight. I only caught it because I compared the printed scores with a hand calculation. That's why I now always ask for a math check after code changes.

---

## Design Pattern (SF10)

> Document how AI helped you choose or implement a design pattern.

**Which design pattern did you use?**

<!-- e.g., Strategy, Factory, Observer, etc. -->

**How did AI help you brainstorm or implement it?**

<!-- Describe the conversation or suggestions that led to your decision -->

**How does the pattern appear in your final code?**

<!-- Point to the relevant class or method -->

---

## Recommendation agent traces

The recommender can plan, call catalog tools, read the results, and then answer. `src/agent.py` exposes `search_catalog`, `filter_songs`, `get_song_details`, and `check_constraints`. Claude uses the Anthropic tools API for that loop (at most 6 steps). With no API key, the same tools run in a fixed order and the deterministic scorer supplies the songs. Each step is shown in the Streamlit "How the AI decided" tab.

Sample traces from offline runs:

- [upbeat happy pop](docs/agent_traces/upbeat-happy-pop.json)
- [family-friendly happy pop](docs/agent_traces/family-friendly-happy-pop-with-no-explic.json)
- [calm instrumental lofi](docs/agent_traces/calm-instrumental-lofi-for-studying.json)

Listening guides in `data/knowledge/` and the offline curator comparison in `eval/style_comparison.txt` are the other stretch notes. The late-night drive before/after is `eval/rag_comparison.txt`.

The happy-pop plan and first tool call:

```json
{
  "step": 0,
  "kind": "plan",
  "text": "No Claude tool call. Plan: search_catalog with the parsed profile, filter explicit songs when asked, check_constraints, then return that ranking."
}
```

```json
{
  "step": 1,
  "kind": "tool",
  "name": "search_catalog",
  "input": {"genre": "pop", "mood": "happy", "energy": 0.85, "allow_explicit": true, "k": 8}
}
```
