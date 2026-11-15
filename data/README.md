# Song catalog

`songs.csv` is what the recommender loads. It has two parts:

1. `starter_songs.csv`: the original 20 fictional songs, kept first and unchanged.
2. 1,560 real songs from the Spotify Tracks Dataset: the 15 most popular songs from each of 104 genres.

Total size is 1,580 songs. `songs.csv` is committed, so the app runs without the raw dataset. To rebuild it:

```bash
python scripts/build_catalog.py --download
```

That downloads the dataset to `data/raw/spotify_tracks.csv` (about 20 MB, gitignored), checks its SHA-256 (`b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd`), and writes `songs.csv`. `--per-genre` sets how many songs each genre keeps (default 15). The output is deterministic: songs are ordered by popularity, then track ID.

## Source and license

- Dataset: "Spotify Tracks Dataset" by Maharshi Pandya, 114,000 tracks across 114 genres with Spotify audio features.
- Download used here: <https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset> (the same data is on Kaggle).
- The Hugging Face page lists the license as BSD. The data was collected from Spotify's Web API, so Spotify's terms may also apply. This project uses it for a non-commercial class assignment, and no audio is copied: the app embeds Spotify's own player.

Spotify's Web API no longer gives new apps access to audio features or recommendations (November 2024), which is why the catalog comes from a dataset collected before that change and not from live API calls.

## What was filtered out

- Genres that are really moods or settings (`sad`, `happy`, `chill`, `party`, `romance`, `sleep`, `study`, `kids`, `children`) and `comedy`. Keeping them would let "happy pop" match the genre `happy` instead of the mood.
- Spoken-word tracks (speechiness ≥ 0.66).
- Duplicates. The same song often appears under several genres; it is kept once, under the first genre (alphabetically) that picks it, and songs that repeat a title and artist are skipped.

## How columns were mapped

| Catalog column | Spotify dataset source |
|---|---|
| title, artist | `track_name`; `artists` with `;` replaced by `, ` |
| genre | `track_genre` with hyphens turned into spaces (`hip-hop` → `hip hop`, `r-n-b` → `r&b`), except names such as `k-pop` and `singer-songwriter` |
| energy, valence, danceability, acousticness, instrumentalness, popularity, explicit | same columns |
| tempo_bpm | `tempo` |
| spotify_id | `track_id`, used for the embedded player and "Open in Spotify" link |
| release_decade | blank: the dataset has no release year, so decade requests only match starter songs |
| language | `none` when instrumentalness ≥ 0.5; a code for genres that name a language (`k-pop` → `ko`, `latin` → `es`, `brazil` → `pt`, `french` → `fr`, and so on); otherwise blank, meaning unknown. A blank language never earns language points. |
| lyrical_theme | `none`: the dataset has no lyric data |

## Mood

The dataset has no mood labels. `scripts/build_catalog.py` derives mood from energy and valence:

- happy: valence ≥ 0.62 and energy ≥ 0.62
- chill: valence ≥ 0.55 and energy < 0.45, or energy ≤ 0.32
- intense: valence < 0.4 and energy ≥ 0.7, or energy ≥ 0.8
- sad: valence < 0.4 and energy < 0.45
- otherwise moody

Mood tags (`energetic`, `mellow`, `bright`, `dark`, `instrumental`) come from the same features. The starter songs keep their hand-written moods.

The `mood_source` column records which is which: `labeled` for the starter songs and `derived` for Spotify tracks. The scorer gives the +1.0 mood bonus only to labeled moods, because a derived mood repeats the energy and valence the scorer already compares.
