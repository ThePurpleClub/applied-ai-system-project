# Song catalog

`songs.csv` is what the recommender loads. It has two parts:

1. `starter_songs.csv` — the original 20 fictional songs, kept first and unchanged.
2. A 391-track sample from the [Free Music Archive (FMA)](https://github.com/mdeff/fma) metadata, joined to Echonest audio features.

Total size is 411 songs. Rebuild it with:

```bash
python scripts/build_catalog.py --download
```

That downloads `fma_metadata.zip` (sha1 `f0df49ffe5f2a6008d7dc83c6915b31835dfe733`), keeps one track title per artist name, balances genres, and writes `songs.csv`. `--limit` defaults to 400. The seed is 14.

## Source and license

FMA was published by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson (EPFL LTS2), paper [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840).

- Metadata: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0).
- Download: <https://os.unil.cloud.switch.ch/fma/fma_metadata.zip>
- The audio files are not included. Each recording stays under the license its artist chose on the Free Music Archive.

## How columns were mapped

| Catalog column | FMA source |
|---|---|
| title, artist | track title, artist name |
| genre | top-level genre, renamed (`Hip-Hop` → `hip hop`, `Electronic` → `edm`, `Soul-RnB` → `r&b`, and so on) |
| energy, valence, danceability, acousticness, instrumentalness, tempo_bpm | Echonest `audio_features` |
| popularity | Echonest `song_hotttnesss` scaled to 0–100 |
| release_decade | track `date_created`, snapped to a decade |
| language | `none` when instrumentalness is at least 0.5, otherwise the track language code |
| lyrical_theme | `none` — FMA does not label themes |
| explicit | `true` only when a track tag contains "explicit" |

FMA does not have Spotify's explicit-lyrics flag, and almost none of the sampled tags say "explicit". The starter songs still carry the explicit labels they were written with, and the scorer's explicit penalty and the "hide explicit" filter use this column.

## Mood

FMA does not label mood. `scripts/build_catalog.py` derives it from energy and valence:

- happy: valence ≥ 0.62 and energy ≥ 0.62
- chill: valence ≥ 0.55 and energy < 0.45, or energy ≤ 0.32
- intense: valence < 0.4 and energy ≥ 0.7, or energy ≥ 0.8
- sad: valence < 0.4 and energy < 0.45
- otherwise moody

Mood tags (`energetic`, `mellow`, `bright`, `dark`, `instrumental`) come from the same features. The starter songs keep their hand-written moods.
