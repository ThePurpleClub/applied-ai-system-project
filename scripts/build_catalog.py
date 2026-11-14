"""Build data/songs.csv from the starter songs plus a Free Music Archive sample.

The FMA metadata is CC BY 4.0. This script downloads fma_metadata.zip when you
pass --download, or reads a zip you already have. It keeps every starter song,
adds up to --limit tracks that have Echonest audio features, and writes the
scorer's column schema.

Mood is not in FMA. It is derived from energy and valence (see data/README.md).
FMA does not label explicit lyrics, so explicit is true only when a track tag
contains "explicit".

Usage:
    python scripts/build_catalog.py --zip /tmp/fma/fma_metadata.zip
    python scripts/build_catalog.py --download
"""

import argparse
import ast
import csv
import hashlib
import io
import math
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STARTER_PATH = ROOT / "data" / "starter_songs.csv"
OUTPUT_PATH = ROOT / "data" / "songs.csv"
FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"

COLUMNS = [
    "id",
    "title",
    "artist",
    "genre",
    "mood",
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
]

GENRE_MAP = {
    "Hip-Hop": "hip hop",
    "Soul-RnB": "r&b",
    "Electronic": "edm",
    "Pop": "pop",
    "Rock": "rock",
    "Jazz": "jazz",
    "Classical": "classical",
    "Blues": "blues",
    "Country": "country",
    "Folk": "folk",
    "International": "international",
    "Experimental": "experimental",
    "Spoken": "spoken",
    "Easy Listening": "easy listening",
    "Instrumental": "instrumental",
    "Old-Time / Historic": "old-time",
}


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


def decade_from(value) -> int:
    if pd.isna(value):
        return 2010
    year = pd.to_datetime(value, errors="coerce")
    if pd.isna(year):
        return 2010
    year = int(year.year)
    if year < 1900 or year > 2029:
        return 2010
    return year - (year % 10)


def key_for(title: str, artist: str) -> str:
    return " ".join(f"{title} {artist}".lower().split())


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_zip_csv(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with archive.open(name) as handle:
        raw = io.BytesIO(handle.read())
    if name.endswith("echonest.csv"):
        return pd.read_csv(raw, index_col=0, header=[0, 1, 2])
    return pd.read_csv(raw, index_col=0, header=[0, 1])


def load_fma(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        tracks = read_zip_csv(archive, "fma_metadata/tracks.csv")
        echo = read_zip_csv(archive, "fma_metadata/echonest.csv")
    audio = echo["echonest"]["audio_features"]
    hot = echo["echonest"]["social_features"]["song_hotttnesss"]
    frame = audio.join(tracks["track"][["title", "genre_top", "date_created", "tags", "language_code", "listens"]])
    frame = frame.join(tracks["artist"]["name"].rename("artist"))
    frame = frame.join(hot.rename("hotttnesss"))
    frame = frame.dropna(subset=["title", "artist", "genre_top", "energy", "valence", "danceability", "acousticness", "instrumentalness", "tempo"])
    frame["genre"] = frame["genre_top"].map(GENRE_MAP)
    frame = frame.dropna(subset=["genre"])
    return frame


def sample_tracks(frame: pd.DataFrame, limit: int, seed: int, reserved: set) -> pd.DataFrame:
    shuffled = frame.sample(frac=1, random_state=seed)
    per_genre = max(1, math.ceil(limit / shuffled["genre"].nunique()))
    chosen = []
    seen = set(reserved)
    for _genre, group in shuffled.groupby("genre", sort=True):
        kept = 0
        for index, row in group.iterrows():
            identity = key_for(str(row["title"]), str(row["artist"]))
            if identity in seen or not str(row["title"]).strip() or not str(row["artist"]).strip():
                continue
            seen.add(identity)
            chosen.append(index)
            kept += 1
            if kept >= per_genre:
                break
    if len(chosen) > limit:
        chosen = list(pd.Series(chosen).sample(limit, random_state=seed))
    return shuffled.loc[chosen]


def row_from_fma(track_id: int, row: pd.Series) -> dict:
    energy = float(row["energy"])
    valence = float(row["valence"])
    instrumentalness = float(row["instrumentalness"])
    tags = []
    if isinstance(row["tags"], str) and row["tags"].startswith("["):
        try:
            tags = [str(tag) for tag in ast.literal_eval(row["tags"])]
        except (SyntaxError, ValueError):
            tags = []
    explicit = any("explicit" in tag.lower() for tag in tags)
    language = "none" if instrumentalness >= 0.5 else str(row["language_code"] or "en").strip().lower() or "en"
    if language in {"nan", "none", ""}:
        language = "none" if instrumentalness >= 0.5 else "en"
    hot = row["hotttnesss"]
    if pd.notna(hot):
        popularity = int(round(min(1.0, max(0.0, float(hot))) * 100))
    else:
        listens = float(row["listens"]) if pd.notna(row["listens"]) else 0
        popularity = int(min(100, round(20 * math.log10(listens + 1))))
    return {
        "id": 100000 + int(track_id),
        "title": " ".join(str(row["title"]).split()),
        "artist": " ".join(str(row["artist"]).split()),
        "genre": row["genre"],
        "mood": derive_mood(energy, valence),
        "energy": round(energy, 4),
        "tempo_bpm": round(float(row["tempo"]), 2),
        "valence": round(valence, 4),
        "danceability": round(float(row["danceability"]), 4),
        "acousticness": round(float(row["acousticness"]), 4),
        "instrumentalness": round(instrumentalness, 4),
        "popularity": popularity,
        "release_decade": decade_from(row["date_created"]),
        "mood_tags": mood_tags(energy, valence, instrumentalness),
        "language": language,
        "explicit": "true" if explicit else "false",
        "lyrical_theme": "none",
    }


def read_starter(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_catalog(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build(zip_path: Path, limit: int, seed: int) -> None:
    starter = read_starter(STARTER_PATH)
    reserved = {key_for(row["title"], row["artist"]) for row in starter}
    frame = load_fma(zip_path)
    sample = sample_tracks(frame, limit, seed, reserved)
    extra = [row_from_fma(track_id, row) for track_id, row in sample.iterrows()]
    extra.sort(key=lambda row: (row["genre"], row["artist"], row["title"]))
    write_catalog(OUTPUT_PATH, starter + extra)
    print(f"Wrote {len(starter)} starter songs and {len(extra)} FMA songs to {OUTPUT_PATH}")


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FMA_URL}")
    urlretrieve(FMA_URL, destination)
    digest = sha1_file(destination)
    if digest != FMA_SHA1:
        raise SystemExit(f"Checksum mismatch: {digest}")
    print("Checksum ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the music catalog CSV.")
    parser.add_argument("--zip", type=Path, help="Path to fma_metadata.zip")
    parser.add_argument("--download", action="store_true", help="Download fma_metadata.zip into --zip")
    parser.add_argument("--limit", type=int, default=400, help="Maximum FMA tracks to add")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    zip_path = args.zip or Path("/tmp/fma/fma_metadata.zip")
    if args.download or not zip_path.exists():
        download(zip_path)
    build(zip_path, args.limit, args.seed)


if __name__ == "__main__":
    main()
