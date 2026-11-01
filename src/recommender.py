import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    instrumentalness: float = 0.0
    popularity: int = 50

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    target_valence: float = 0.65  # dataset average, i.e. neutral
    prefers_instrumental: Optional[bool] = None  # None = no preference
    prefers_popular: Optional[bool] = None  # None = no preference

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs from a CSV file into a list of dicts, converting numeric columns to int/float."""
    int_fields = ["id", "popularity"]
    float_fields = ["energy", "tempo_bpm", "valence", "danceability", "acousticness", "instrumentalness"]

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # DictReader returns every value as a string, so convert the numeric columns
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)

    print(f"Loaded {len(songs)} songs from {csv_path}")
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against the user's preferences (max 5.0) and return (score, reasons)."""
    score = 0.0
    reasons = []

    # Genre: full points for an exact match, half for a related genre (e.g. "pop" vs "indie pop")
    user_genre = user_prefs["genre"].lower()
    song_genre = song["genre"].lower()
    if song_genre == user_genre:
        score += 2.0
        reasons.append("genre match (+2.0)")
    elif user_genre in song_genre.split() or song_genre in user_genre.split():
        score += 1.0
        reasons.append(f"related genre: {song['genre']} (+1.0)")

    if song["mood"].lower() == user_prefs["mood"].lower():
        score += 1.0
        reasons.append("mood match (+1.0)")

    # Similarity points: closer to the target means more points
    energy_points = 1.5 * (1 - abs(song["energy"] - user_prefs["energy"]))
    score += energy_points
    reasons.append(f"energy {song['energy']:.2f} vs target {user_prefs['energy']:.2f} (+{energy_points:.2f})")

    target_valence = user_prefs.get("valence", 0.65)  # dataset average, i.e. neutral
    valence_points = 0.5 * (1 - abs(song["valence"] - target_valence))
    score += valence_points
    reasons.append(f"valence {song['valence']:.2f} vs target {target_valence:.2f} (+{valence_points:.2f})")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song and return the top k as (song, score, explanation), highest score first."""
    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:k]
    return [(song, score, ", ".join(reasons)) for song, score, reasons in ranked]
