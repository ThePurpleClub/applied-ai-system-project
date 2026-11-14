import csv
from pathlib import Path

from src.song_map import project_profile, project_songs

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_keeps_starter_songs_and_adds_fma_tracks():
    rows = list(csv.DictReader((ROOT / "data" / "songs.csv").open(encoding="utf-8")))
    starter = list(csv.DictReader((ROOT / "data" / "starter_songs.csv").open(encoding="utf-8")))

    assert len(rows) >= 300
    assert rows[: len(starter)] == starter
    assert len({row["id"] for row in rows}) == len(rows)
    assert "happy" in {row["mood"] for row in rows}
    assert "true" in {row["explicit"] for row in rows}
    assert (ROOT / "data" / "songs.csv").stat().st_size < 2_000_000


def test_projection_is_stable_and_places_a_profile():
    songs = [
        {
            "id": 1, "title": "Bright", "artist": "A", "genre": "pop", "mood": "happy",
            "energy": 0.9, "valence": 0.8, "danceability": 0.7, "acousticness": 0.1,
            "instrumentalness": 0.0, "tempo_bpm": 120,
        },
        {
            "id": 2, "title": "Soft", "artist": "B", "genre": "lofi", "mood": "chill",
            "energy": 0.2, "valence": 0.4, "danceability": 0.3, "acousticness": 0.9,
            "instrumentalness": 0.8, "tempo_bpm": 70,
        },
        {
            "id": 3, "title": "Drive", "artist": "C", "genre": "rock", "mood": "intense",
            "energy": 0.7, "valence": 0.3, "danceability": 0.5, "acousticness": 0.2,
            "instrumentalness": 0.1, "tempo_bpm": 150,
        },
    ]
    model = project_songs(songs)
    again = project_songs(songs)

    assert len(model["points"]) == 3
    assert [point["x"] for point in model["points"]] == [point["x"] for point in again["points"]]
    placed = project_profile({"energy": 0.8, "valence": 0.7}, model)
    assert placed is not None
    assert len(placed) == 2
    assert all(value == value for value in placed)
