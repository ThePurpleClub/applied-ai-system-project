# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**VibeMatch 1.0**, a content-based music recommender that matches songs to a listener's genre, mood and energy.

---

## 2. Intended Use  

VibeMatch takes one listener's taste profile and returns the 5 songs from a small catalog that fit it best, each with a short explanation of why it was picked. It's built for classroom exploration, to learn how a scoring-based recommender works and where it goes wrong. It isn't meant for real users.

It assumes a lot about the listener:
- **One taste at a time.** It assumes the listener has a single favorite genre, a single mood, and a fixed energy level they want right now.
- **Exact labels.** It assumes those preferences can be described with the same words the catalog uses.
- **"Not specified" means "medium".** If the listener doesn't say how upbeat they want the music, it assumes a medium value instead of no preference (see Limitations and Bias).

It doesn't learn from listening history, skips or other users. It only knows what's in the profile.

---

## 3. How the Model Works  

Think of the recommender as a judge at a talent show. It looks at every song in the catalog, one at a time, gives each one a score out of 5 based on how well it fits what you like, and then shows you the five highest-scoring songs.

**What it knows about each song.** Every song has a genre (like lofi or rock), a mood (like chill or intense), an energy level from 0 (very calm) to 1 (very intense), and a valence level from 0 (sad or dark) to 1 (happy or bright). The catalog also lists tempo, danceability, acousticness and other details, but the scoring ignores those. Fast songs are usually high-energy songs, so counting tempo too would mostly reward the same thing twice.

**What it knows about you.** Your taste profile is your favorite genre, your favorite mood, how energetic you want the music to be, and optionally how upbeat you want it to be. If you don't say how upbeat, it assumes you're in the middle. The profile also has "likes acoustic", "prefers instrumental" and "prefers popular" settings, but the scoring doesn't use them yet.

**How a song earns points.**

- **Genre (up to 1.5 points).** A song in your favorite genre gets 1.5 points. A song in a closely related genre, like "indie pop" when you like "pop", gets 0.75. Genre is worth more than mood because it's the more stable part of someone's taste: a metal fan rarely wants a lofi song, even on a calm day.
- **Mood (1 point).** A song with exactly your mood gets 1 point. Mood is worth less than genre because it changes from day to day, and because the energy and valence scores already reflect it. Only hand-written moods earn this point. The Spotify songs have moods that were *computed* from their energy and valence, so giving them a mood point would reward the same two numbers twice. Their explanation still says the mood matches, with no bonus.
- **Energy (up to 2 points).** Rather than "more energy is better", the closer a song's energy is to the energy you asked for, the more points it gets. A perfect match earns all 2 points, a song that's a little off earns a bit less, and a song at the opposite extreme earns nothing.
- **Valence (up to 0.5 points).** This works the same way as energy but counts for less. It mostly acts as a tiebreaker between songs that are otherwise very similar.

Add those up and the best possible song scores 5. Energy (up to 2) can outweigh a genre match on its own (1.5), so a slow song in your genre won't beat a song with the energy and mood you asked for. But matching genre *and* mood (2.5) still beats energy. I picked these weights after testing two other versions (see Evaluation). The original weights were genre 2 and energy 1.5, and they let genre override everything else. Every time a song earns points, the system writes down why (for example, "mood match +1.0"), so each recommendation comes with a short explanation.

**What I changed from the starter.** The starter code didn't score anything: it loaded no songs and returned no recommendations, and the class version simply handed back the first few songs in the file. I added the point system above, partial credit for related genres, and the valence tiebreaker, and I added the "why" explanations so you can see exactly how each score was earned. After testing edge cases, I also made it ignore extra spaces and capital letters in genre and mood, cap energy and valence targets to the 0–1 range so scores can't go negative, and fill in defaults when a profile leaves something out instead of crashing.

---

## 4. Data  

The catalog is `data/songs.csv`, with **1,580 songs** by 1,124 artists: the 20 fictional starter songs (also kept in `data/starter_songs.csv`) and 1,560 real songs from the Spotify Tracks Dataset (the 15 most popular songs from each of 104 genres). `scripts/build_catalog.py` builds it, and `data/README.md` lists the source, license note, filters and column mapping.

**What I changed:** the starter file had 10 songs. I added 10 more to cover genres it was missing (metal, classical, country, r&b, hip hop, reggae, edm, bluegrass, dream pop and blues), and I added two columns, `instrumentalness` and `popularity`. Later I added decade, language, mood-tag, explicit and theme columns, a `mood_source` column that says whether a mood was written by hand (`labeled`) or computed (`derived`), and real songs with a `spotify_id` so the app can play them.

**What it contains:**
- **Genres:** 107 genres. Each of the 104 Spotify genres has exactly 15 songs. Lofi (3 songs), synthwave and dream pop exist only among the starter songs.
- **Moods:** the Spotify moods come from a rule on energy and valence, so they use only five labels: moody (537 songs), intense (383), happy (378), sad (196) and chill (66). Sixteen moods appear among the starter songs.
- **Energy:** ranges from 0 to 1 and leans loud: 791 songs are above 0.7 and 230 are below 0.4.
- **Valence:** ranges from 0.02 to 0.98 with an average of 0.49. 634 songs are below 0.4.
- **Popularity:** the Spotify songs are each genre's most popular, with a median popularity of 73 out of 100.
- **Explicit:** 199 songs are marked explicit, using Spotify's own flag.

**What's missing:**
- **Real mood labels:** the dataset doesn't label mood, so 1,560 of the 1,580 moods are a guess from two numbers. A slow, bright jazz song and a slow, bright folk song get the same mood even if they feel nothing alike.
- **Release year:** the dataset has no year, so decade requests ("80s music") only match starter songs.
- **Language:** only songs in language-named genres (k-pop, latin, brazil, french and so on) or instrumental songs have a language. Most English songs have a blank language, so "songs in English" only matches English starter songs.
- **Hits only:** each genre keeps only its 15 most popular songs, so lesser-known artists are missing.
- **Single labels:** each song has one genre and one mood. Real songs are often several of each.
- **The "average" isn't the average:** the code describes its default valence of 0.65 as "the dataset average". That was close for the 20 starter songs (0.60) and is off for the current catalog (0.49).

---

## 5. Strengths  

- **Listeners whose preferences agree with each other.** When the genre, mood and energy all point the same way, the #1 pick is the obvious one: Sunrise City for the pop fan, Library Rain for the lofi fan, Storm Runner for the rock fan. In my weight experiments, the #1 song never changed for any profile, so these results don't depend on the exact numbers I chose.
- **Genres with several songs.** The Chill Lofi listener gets the best list: three lofi songs, then two other calm songs (ambient and jazz) that a lofi fan would plausibly enjoy.
- **Closeness instead of "more is better".** Scoring energy by how *close* a song is to the target means a calm listener isn't pushed toward louder songs, and the pop and lofi lists share no songs at all.
- **Songs that fit two listeners for different reasons.** Gym Hero (pop, intense) shows up for both the pop fan, who gets the genre points, and the rock fan, who gets the mood point. That matches intuition: it's a pop song that sounds like a workout track.
- **Every score is explained.** Each recommendation lists exactly where its points came from, and the numbers add up to the score. That made it easy to see *why* something odd was ranked high, which is how I found most of the problems in this card.
- **It handles messy input.** Extra spaces, capital letters, an energy value of 1.5, or a missing mood no longer change the results or crash the program.

---

## 6. Limitations and Bias 

The system quietly works against sad and dark music, even for users who never said they dislike it. When a profile doesn't set a valence, the scorer assumes a "neutral" target of 0.65, the dataset average. So every song is still judged on how close it is to 0.65, and dark songs like Iron Cathedral (0.22), Nocturne in Grey (0.21) and Last Train Blues (0.25) lose about 0.2 points for every one of those users. When I scored 1,360 simulated users covering every genre, mood and energy combination, Iron Cathedral and Last Train Blues were the two least-recommended songs in the catalog (127 and 147 top-5 appearances), while Spacewalk Thoughts (valence exactly 0.65) made it 621 times, against a fair share of about 340. A real fix would be to skip the valence score entirely when the user hasn't stated a preference, instead of treating "no opinion" as "medium-happy." Those numbers come from the original 20-song catalog. The current catalog's average valence is 0.49 and 634 of its 1,580 songs are below 0.4, so the 0.65 default pushes against a large share of the catalog.

**Starter songs get an advantage on mood requests.** Only the 20 hand-written songs have real mood labels, so only they can earn the mood point. In a spot check of seven fallback requests, starter songs took 11 of the 35 top-5 slots even though they are about 1% of the catalog (*Sunrise City*, *Gym Hero* and *Last Train Blues* show up often). Giving computed moods the point would remove that advantage, but every real song with a matching computed mood would then earn extra points for energy and valence it had already been scored on. I chose the version that doesn't double count, and I'm treating the starter-song advantage as a known bias. A fairer fix would be real mood labels for the Spotify tracks.

---

## 7. Evaluation  

**Profiles I tested.** All of the results in this section were measured on the original 20-song catalog, before the real songs were added. I ran nine taste profiles through `python -m src.main`. Three are typical listeners: *High-Energy Pop* (pop, happy, energy 0.85), *Chill Lofi* (lofi, chill, energy 0.38) and *Deep Intense Rock* (rock, intense, energy 0.92). The other six are edge cases built to try to trick the scoring:

- *Sad Banger*: sad blues but energy 0.9, so the preferences contradict each other.
- *Genre vs Vibe*: classical, but euphoric with energy 0.95, so genre pulls one way and everything else pulls the other.
- *Messy Input*: genre `" Pop "` and mood `"HAPPY"`, with extra spaces and odd capitals.
- *Out of Range Energy*: energy 1.5, outside the 0–1 scale.
- *Unknown Genre*: k-pop, which isn't in the catalog.
- *No Signal*: a genre and mood that match nothing, with every number at 0.5.

**What I looked for.** For the typical profiles, I checked whether the top songs were ones a real fan would pick, and whether the "why" explanations made sense. For the edge cases, I looked for crashes, negative scores, scores above the maximum, and rankings where one preference silently overrode everything else.

**What worked.** All three typical profiles got a sensible #1 (Sunrise City, Library Rain, Storm Runner). Chill Lofi got the cleanest list: three lofi songs, then two other calm songs.

**What I found and fixed.** The edge cases found three real bugs:

- *Messy Input:* extra spaces turned an exact pop match into only a "related genre" match, which cost Sunrise City a full point.
- *Out of Range Energy:* energy 1.5 made the energy score go negative.
- *Missing mood:* a profile with no mood crashed the program.

I fixed all three by trimming spaces, capping targets at 0–1 and adding defaults.

**What surprised me.**

- *Contradictory preferences:* genre used to overrule everything. In *Genre vs Vibe*, a slow, melancholic classical piece (Nocturne in Grey, energy 0.18) still came 2nd for someone asking for euphoric, high-energy music.
- *Sad Banger:* the slow blues song wins even now, because matching genre and mood is worth more than matching energy.
- *Valence default:* I didn't expect the default valence to penalize dark songs (see Limitations and Bias).

**Comparing profiles side by side.** For each pair, I looked at what changed between the two top-5 lists and whether the change makes sense. Scores use the current weights (genre 1.5, mood 1.0, energy 0–2.0, valence 0–0.5).

*The three typical listeners:*

- **High-Energy Pop vs Chill Lofi:** the two lists share no songs at all. Pop gets bright, loud tracks around energy 0.8–0.95 (Sunrise City, Gym Hero, Laser Bloom), while Lofi gets quiet tracks between 0.28 and 0.42 (Library Rain, Midnight Coding, Spacewalk Thoughts). This makes sense because the energy targets (0.85 vs 0.38) are almost opposite. A pop song loses about 1 point on energy for the lofi listener, and the genre and mood don't match either.
- **High-Energy Pop vs Deep Intense Rock:** both listeners want loud music, so three songs appear in both lists: Gym Hero, Laser Bloom and Banjo Breakdown. Gym Hero is the interesting one. It's a *pop* song with an *intense* mood, so it earns the genre point for the pop fan and the mood point for the rock fan, and it ranks in the top 3 for both. The difference is valence. The rock fan wants darker music (0.35), so Iron Cathedral (metal, valence 0.22) makes their list, while the pop fan (valence 0.80) gets the brighter Rooftop Lights instead.
- **Chill Lofi vs Deep Intense Rock:** again no overlap, and the gap is even bigger than pop vs lofi, because rock's target energy (0.92) is farther from lofi's (0.38). The lofi list has three songs in its own genre, but the rock list has only one (Storm Runner), because the catalog has 3 lofi songs and just 1 rock song. After Storm Runner, the rock list is filled by loud songs from other genres. That's correct for the energy target, but it shows the catalog limits the rock fan more than the scoring does.

*Each edge case vs the profile it's closest to:*

- **Sad Banger vs Deep Intense Rock:** both want energy around 0.9 and dark valence, and from #2 down the lists are almost the same: Iron Cathedral, Storm Runner, Gym Hero, Laser Bloom. The only real difference is #1. The rock fan gets Storm Runner (energy 0.91), but Sad Banger gets Last Train Blues, a slow blues song with energy 0.40. Last Train Blues matches genre *and* mood (2.5 points), which beats the 1 point it loses for being 0.5 off on energy. This follows the weights correctly, but a listener who mainly wanted energy would be disappointed. When preferences conflict, matching labels win over the energy target.
- **Genre vs Vibe vs High-Energy Pop:** both want loud, happy-sounding music (energy 0.95 vs 0.85, valence 0.90 vs 0.80), so four of five songs overlap. The difference is that Genre vs Vibe asks for *classical*, and the only classical song (Nocturne in Grey, energy 0.18) is now nowhere in the top 5. Instead, Laser Bloom ranks first because it matches the mood (euphoric) and the energy. That makes sense: one genre match shouldn't outweigh a song being the opposite of everything else asked for. With the original weights (genre worth 2.0), Nocturne in Grey was 2nd.
- **Messy Input vs High-Energy Pop:** the top 3 are the same songs in the same order (Sunrise City 4.87 vs 4.92), so typing `" Pop "` and `"HAPPY"` now works just like `"pop"` and `"happy"`. The small differences come from Messy Input's energy target being 0.8 instead of 0.85, and from its valence defaulting to 0.65 because none was given. That's why slot 4 is Night Drive Loop (energy 0.75) instead of Laser Bloom (energy 0.95). Before I fixed the input handling, Sunrise City scored a full point lower here. This pair shows the fix works.
- **Unknown Genre (k-pop) vs Messy Input (pop):** this is the cleanest test, because the two profiles have the same mood and energy and differ *only* in genre. Sunrise City drops by exactly 1.5 (4.87 → 3.37), which is the genre point it lost. Rooftop Lights drops by 0.75 (the related-genre credit). Gym Hero, which only ranked well because it's pop, falls out of the top 5. Everything else is unchanged. This shows that each rule adds exactly the points it should and nothing else. It also shows a gap: "k-pop" gets no credit from pop songs at all.
- **Out of Range Energy (metal, 1.5) vs Deep Intense Rock:** both lists are filled with the same loud songs (Laser Bloom, Gym Hero, Storm Runner, Banjo Breakdown) after the #1 genre match. Energy 1.5 is now treated as 1.0, which is what a user typing 1.5 most likely meant ("as loud as possible"). Iron Cathedral scores 4.72 with no negative or strange numbers. Before the fix, every song with energy below 0.5 got *negative* energy points, because it was more than 1.0 away from the target.
- **No Signal vs Chill Lofi:** two lofi songs (Midnight Coding, Focus Flow) appear in both lists, but for completely different reasons. For the lofi fan, they earn genre points. For No Signal, they just happen to be near energy 0.5. The bigger difference is the score range. Chill Lofi's list goes from 4.94 down to 2.42, while No Signal's is squeezed between 2.21 and 2.37, so the system is basically guessing. That makes sense when nothing matches, but the recommender still presents those five songs with the same confidence as a real match, which is misleading.

**Comparisons I ran.**

1. *Weight experiments.* I compared three weightings on all nine profiles:
   - *Original:* genre 2.0, energy 1.5.
   - *Double energy, half genre:* genre 1.0, energy 3.0.
   - *Middle ground:* genre 1.5, energy 2.0.

   The #1 song never changed. Double energy removed the classical song from *Genre vs Vibe*, but it let any loud song, even bluegrass, into the pop and rock lists, and it squeezed all scores close together. I kept the middle ground.
2. *Math check.*
   - A song that matches a profile perfectly scores exactly 5.0.
   - All 180 song × profile scores fall between 0.64 and 4.94.
   - The points listed in each explanation add up to the song's score.
   - A hand-worked case (Sad Banger vs Last Train Blues = 3.975) matches the code.
3. *Exposure test.* I scored 1,360 simulated users, one for every genre × mood × 5 energy levels, and counted how often each song reached a top 5. A fair share would be about 340 each. The most-shown song (Spacewalk Thoughts, 621) appeared about 5× as often as the least-shown (Iron Cathedral, 127).
4. *Unit tests.* When I wrote this evaluation, `pytest` had 2 tests that only checked a placeholder `Recommender` class. The suite now has 36 tests covering the scorer (including the no-bonus rule for computed moods), the Claude pipeline with a mocked client, refinement and the self-check, the Song Map projection, and the Streamlit page.

---

## 8. Future Work  

- **Stop treating "no opinion" as a preference.** If a listener doesn't give a valence, skip the valence score instead of comparing every song to 0.65. This one change removes the bias against sad and dark songs described in Limitations and Bias.
- **Use the preferences the profile already has.** `likes_acoustic`, `prefers_instrumental` and `prefers_popular` are collected but ignored, so a listener who asks for acoustic music can still get EDM. Each could be a small bonus based on the `acousticness`, `instrumentalness` and `popularity` columns. Popularity should be added carefully, because in this catalog it also rewards loud songs.
- **Smarter genre and mood matching.** Replace "the two genre names share a word" with a short hand-written list of related genres, so rock ↔ metal and blues ↔ jazz get partial credit, and "k-pop" counts as close to pop. Do the same for moods, so "sad" ↔ "melancholic" and "chill" ↔ "relaxed" earn partial points.
- **More variety in the top 5.** Limit each list to one song per artist, or give a small penalty to a song that's very similar to one already picked. That stops the list from filling with near-duplicates and gives less-shown songs a fair chance.
- **Admit when nothing fits.** If even the best song scores low (like the No Signal profile, where every song scored about 2.3 out of 5), say "no strong matches" instead of presenting five guesses with the same confidence as real matches.
- **Handle more complex tastes.** Allow more than one favorite genre or mood, a range of energy ("between 0.4 and 0.7") instead of one number, and different profiles for different situations (studying, workout).
- **A bigger, more balanced catalog.** Several songs per genre, more sad and low-energy songs, and music beyond mostly Western genres, so a fan of a niche genre gets more than one real match.
- **Real mood labels, years and languages for the Spotify tracks.** Computed moods can't earn points without double counting, so starter songs win mood requests, and the missing release years and languages mean decade and language requests only match starter songs. Tagging the Spotify tracks by hand, or with Claude from their titles and genres, would let every song compete on mood.

---

## 9. Personal Reflection  

**What I learned about recommender systems.** A recommender is mostly a list of choices about what matters and how much. Mine gives a song points for matching genre, mood, energy and valence, and the "weights" decide which of those wins when they disagree. Changing just two numbers (how much genre and energy are worth) changed which songs showed up for almost every listener, even though the code barely changed. The weights *are* the opinion the system has about music.

**Something unexpected.** The biggest problems weren't in the obvious places. I expected bugs from strange inputs, and I found some (extra spaces, an energy of 1.5, a missing mood). But the most surprising problem came from a default value. When a listener didn't say how happy or sad they wanted their music, the system quietly assumed "medium", and that made dark songs like Iron Cathedral and Last Train Blues some of the least-recommended songs in the catalog, even though nobody asked for that. A choice that looked harmless ended up being unfair to a whole kind of music.

**How this changed how I think about music apps.** When an app like Spotify recommends something, I now think about what it's assuming about me that I never told it, and which songs it rarely shows anyone. If my 20-song system already had songs shown five times as often as others, a real app with millions of songs and listeners can create much bigger filter bubbles. Explanations helped a lot here: because every score showed where its points came from, I could actually see *why* the system made a weird choice. I'd like real apps to show more of that.

**Working with AI.** I used an AI assistant to help write test profiles, run experiments and check the math. It was most useful for finding problems quickly, but it also made mistakes (for example, one edit accidentally changed the wrong weights), so I had to check its results against the actual output instead of trusting them.  
