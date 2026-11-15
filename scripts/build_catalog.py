"""Build data/songs.csv from the starter songs plus a Spotify Tracks Dataset sample.

Source: the "Spotify Tracks Dataset" by Maharshi Pandya (114,000 tracks with
Spotify audio features), mirrored on Hugging Face. See data/README.md for the
license note. Pass --download to fetch it into data/raw/, or --csv to use a
copy you already have.

The script keeps every starter song, then takes the most popular songs from
each genre so every genre is represented evenly. It writes the scorer's column
schema plus spotify_id, which the app uses for the embedded Spotify player.

The dataset has no mood, release year, language, or lyric theme:
- mood is derived from energy and valence (mood_source "derived", no bonus).
- release_decade is left blank, so decade requests only match starter songs.
- language is "none" for instrumental tracks, a code for genres that name a
  language (k-pop -> ko, for example), and blank otherwise.

Usage:
    python scripts/build_catalog.py --download
    python scripts/build_catalog.py --csv data/raw/spotify_tracks.csv
"""

import argparse
import csv
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STARTER_PATH = ROOT / "data" / "starter_songs.csv"
OUTPUT_PATH = ROOT / "data" / "songs.csv"
RAW_PATH = ROOT / "data" / "raw" / "spotify_tracks.csv"
DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
DATASET_SHA256 = "b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd"

COLUMNS = [
    "id",
    "title",
    "artist",
    "genre",
    "mood",
    "mood_source",
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "popularity",
    "release_decade",
    "mood_tags",
    "language",
    "explicit",
    "lyrical_theme",
    "spotify_id",
]

# Dataset "genres" that are really moods or settings. Keeping them would let a
# request such as "happy pop" match the genre "happy" instead of the mood.
SKIP_GENRES = {"comedy", "sad", "happy", "chill", "party", "romance", "sleep", "study", "kids", "children"}

GENRE_NAMES = {
    "r-n-b": "r&b",
    "rock-n-roll": "rock and roll",
    "drum-and-bass": "drum and bass",
    # Hyphens that are part of the name stay.
    "k-pop": "k-pop",
    "j-pop": "j-pop",
    "j-rock": "j-rock",
    "j-idol": "j-idol",
    "j-dance": "j-dance",
    "singer-songwriter": "singer-songwriter",
}

GENRE_LANGUAGE = {
    "k-pop": "ko",
    "j-pop": "ja",
    "j-rock": "ja",
    "j-idol": "ja",
    "j-dance": "ja",
    "anime": "ja",
    "mandopop": "zh",
    "cantopop": "zh",
    "spanish": "es",
    "latin": "es",
    "latino": "es",
    "reggaeton": "es",
    "salsa": "es",
    "tango": "es",
    "brazil": "pt",
    "mpb": "pt",
    "sertanejo": "pt",
    "pagode": "pt",
    "forro": "pt",
    "samba": "pt",
    "french": "fr",
    "german": "de",
    "swedish": "sv",
    "turkish": "tr",
    "iranian": "fa",
    "malay": "ms",
}


def genre_name(raw: str) -> str:
    """Spotify genre slug to catalog genre, e.g. hip-hop -> hip hop."""
    return GENRE_NAMES.get(raw, raw.replace("-", " "))


def derive_mood(energy: float, valence: float) -> str:
    """Map audio features onto the scorer's mood labels."""
    if valence >= 0.62 and energy >= 0.62:
        return "happy"
    if valence >= 0.55 and energy < 0.45:
        return "chill"
    if valence < 0.4 and energy >= 0.7:
        return "intense"
    if valence < 0.4 and energy < 0.45:
        return "sad"
    if energy >= 0.8:
        return "intense"
    if energy <= 0.32:
        return "chill"
    return "moody"


def mood_tags(energy: float, valence: float, instrumentalness: float) -> str:
    tags = []
    if energy >= 0.75:
        tags.append("energetic")
    if energy <= 0.4:
        tags.append("mellow")
    if valence >= 0.7:
        tags.append("bright")
    if valence <= 0.35:
        tags.append("dark")
    if instrumentalness >= 0.5:
        tags.append("instrumental")
    return "|".join(tags)


def song_key(title: str, artist: str) -> str:
    return " ".join(f"{title} {artist}".lower().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame = frame.dropna(subset=["track_id", "track_name", "artists"])
    frame = frame[~frame["track_genre"].isin(SKIP_GENRES)]
    # Spoken word (stand-up, audiobooks) is not music the scorer can describe.
    frame = frame[frame["speechiness"] < 0.66]
    frame = frame.copy()
    frame["artist"] = frame["artists"].str.split(";").map(lambda names: ", ".join(name.strip() for name in names))
    frame["title"] = frame["track_name"].map(lambda text: " ".join(str(text).split()))
    return frame


def sample_tracks(frame: pd.DataFrame, per_genre: int, reserved: set) -> pd.DataFrame:
    """Take the most popular unseen songs from each genre, so genres stay balanced.

    The same song often appears under several genres. It is kept once, under
    the first genre (alphabetically) that picks it.
    """
    frame = frame.sort_values(["popularity", "track_id"], ascending=[False, True])
    seen_ids, seen_keys = set(), set(reserved)
    picked = []
    for genre in sorted(frame["track_genre"].unique()):
        count = 0
        for _, row in frame[frame["track_genre"] == genre].iterrows():
            key = song_key(row["title"], row["artist"])
            if row["track_id"] in seen_ids or key in seen_keys:
                continue
            seen_ids.add(row["track_id"])
            seen_keys.add(key)
            picked.append(row)
            count += 1
            if count >= per_genre:
                break
    return pd.DataFrame(picked)


def row_from_track(index: int, row) -> dict:
    energy = float(row["energy"])
    valence = float(row["valence"])
    instrumentalness = float(row["instrumentalness"])
    raw_genre = row["track_genre"]
    if instrumentalness >= 0.5:
        language = "none"
    else:
        language = GENRE_LANGUAGE.get(raw_genre, "")
    return {
        "id": 200000 + index,
        "title": row["title"],
        "artist": row["artist"],
        "genre": genre_name(raw_genre),
        "mood": derive_mood(energy, valence),
        "mood_source": "derived",
        "energy": round(energy, 4),
        "tempo_bpm": round(float(row["tempo"]), 2),
        "valence": round(valence, 4),
        "danceability": round(float(row["danceability"]), 4),
        "acousticness": round(float(row["acousticness"]), 4),
        "instrumentalness": round(instrumentalness, 4),
        "popularity": int(row["popularity"]),
        "release_decade": "",
        "mood_tags": mood_tags(energy, valence, instrumentalness),
        "language": language,
        "explicit": "true" if bool(row["explicit"]) else "false",
        "lyrical_theme": "none",
        "spotify_id": row["track_id"],
    }


def read_starter(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_catalog(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build(csv_path: Path, per_genre: int) -> None:
    starter = read_starter(STARTER_PATH)
    reserved = {song_key(row["title"], row["artist"]) for row in starter}
    sample = sample_tracks(load_dataset(csv_path), per_genre, reserved)
    extra = [row_from_track(index, row) for index, (_, row) in enumerate(sample.iterrows(), start=1)]
    write_catalog(OUTPUT_PATH, starter + extra)
    print(f"Wrote {len(starter)} starter songs and {len(extra)} Spotify songs to {OUTPUT_PATH}")


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_URL}")
    urlretrieve(DATASET_URL, destination)
    digest = sha256_file(destination)
    if digest != DATASET_SHA256:
        raise SystemExit(f"Checksum mismatch: {digest}")
    print("Checksum ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the music catalog CSV.")
    parser.add_argument("--csv", type=Path, default=RAW_PATH, help="Path to the Spotify Tracks Dataset CSV")
    parser.add_argument("--download", action="store_true", help="Download the dataset into --csv first")
    parser.add_argument("--per-genre", type=int, default=15, help="Songs to keep from each genre")
    args = parser.parse_args()
    if args.download or not args.csv.exists():
        download(args.csv)
    build(args.csv, args.per_genre)


if __name__ == "__main__":
    main()
