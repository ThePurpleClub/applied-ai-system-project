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
    assert len(placed) == 3
    assert all("z" in point for point in model["points"])
    assert len(model["axis_labels"]) == 3
    assert "↑" in model["axis_labels"][0] or "↓" in model["axis_labels"][0]
    assert all(value == value for value in placed)


def test_profile_marker_uses_retrieved_songs_for_unknown_features():
    songs = [
        {
            "id": index, "title": f"S{index}", "artist": "A", "genre": "pop", "mood": "happy",
            "energy": energy, "valence": 0.5, "danceability": dance, "acousticness": acoustic,
            "instrumentalness": 0.1, "tempo_bpm": tempo,
        }
        for index, (energy, dance, acoustic, tempo) in enumerate(
            [(0.9, 0.9, 0.1, 150), (0.85, 0.8, 0.15, 140), (0.2, 0.2, 0.9, 70), (0.25, 0.3, 0.8, 75)]
        )
    ]
    model = project_songs(songs)
    near = songs[:2]
    placed = project_profile({"energy": 0.88, "valence": 0.5}, model, reference_songs=near)
    default = project_profile({"energy": 0.88, "valence": 0.5}, model)

    def gap(xy, song):
        point = next(p for p in model["points"] if p["id"] == song["id"])
        return sum((a - b) ** 2 for a, b in zip(xy, (point["x"], point["y"], point["z"]))) ** 0.5

    assert gap(placed, near[0]) < gap(default, near[0])


def test_spotify_rows_have_track_ids_and_starter_rows_do_not():
    import re

    rows = list(csv.DictReader((ROOT / "data" / "songs.csv").open(encoding="utf-8")))
    starter_ids = {row["id"] for row in csv.DictReader((ROOT / "data" / "starter_songs.csv").open(encoding="utf-8"))}
    spotify = [row for row in rows if row["id"] not in starter_ids]

    assert len(spotify) >= 1000
    assert all(re.fullmatch(r"[A-Za-z0-9]{22}", row["spotify_id"]) for row in spotify)
    assert all(row["spotify_id"] == "" for row in rows if row["id"] in starter_ids)
    assert all(row["mood_source"] == "derived" and row["release_decade"] == "" for row in spotify)
    assert len({row["spotify_id"] for row in spotify}) == len(spotify)
