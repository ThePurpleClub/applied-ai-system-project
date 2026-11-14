import csv
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("recommender")

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
    release_decade: int = 2020
    mood_tags: str = ""  # detailed tags separated by "|", e.g. "nostalgic|warm"
    language: str = "en"  # "none" for instrumental tracks
    explicit: bool = False
    lyrical_theme: str = "none"

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


def song_to_dict(song: Song) -> Dict:
    """Adapt a Song to the dict shape score_song already expects."""
    return {
        "id": song.id,
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
        "instrumentalness": song.instrumentalness,
        "popularity": song.popularity,
        "release_decade": song.release_decade,
        "mood_tags": song.mood_tags,
        "language": song.language,
        "explicit": song.explicit,
        "lyrical_theme": song.lyrical_theme,
    }


def user_profile_to_prefs(user: UserProfile) -> Dict:
    """Map the class-based profile onto the scorer's preference keys.

    likes_acoustic, prefers_instrumental, and prefers_popular stay on
    UserProfile because the starter tests construct them. score_song does
    not award points for those fields, so they are intentionally not passed
    through. Removing them would change the constructor the tests call.
    """
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "valence": user.target_valence,
    }


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs for this user, highest score_song result first."""
        prefs = user_profile_to_prefs(user)
        scored = []
        for song in self.songs:
            score, _reasons = score_song(prefs, song_to_dict(song))
            scored.append((score, song))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [song for _score, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain one song with the same reasons score_song uses."""
        _score, reasons = score_song(user_profile_to_prefs(user), song_to_dict(song))
        if not reasons:
            return "No preference matches."
        return ", ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs from a CSV file into a list of dicts, converting numeric columns to int/float."""
    int_fields = ["id", "popularity", "release_decade"]
    float_fields = ["energy", "tempo_bpm", "valence", "danceability", "acousticness", "instrumentalness"]

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # DictReader returns every value as a string, so convert the numeric columns
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            row["explicit"] = row["explicit"].strip().lower() == "true"
            songs.append(row)

    logger.info("load: %d songs from %s", len(songs), csv_path)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against the user's preferences (see max_score) and return (score, reasons)."""
    score = 0.0
    reasons = []

    # Genre: full points for an exact match, half for a related genre (e.g. "pop" vs "indie pop")
    user_genre = user_prefs.get("genre", "").strip().lower()
    song_genre = song["genre"].strip().lower()
    if song_genre == user_genre:
        score += 1.5
        reasons.append("genre match (+1.5)")
    elif user_genre in song_genre.split() or song_genre in user_genre.split():
        score += 0.75
        reasons.append(f"related genre: {song['genre']} (+0.75)")

    if song["mood"].strip().lower() == user_prefs.get("mood", "").strip().lower():
        score += 1.0
        reasons.append("mood match (+1.0)")

    # Similarity points: closer to the target means more points.
    # Clamp targets to 0-1 so out-of-range input can't produce negative points.
    target_energy = min(max(user_prefs.get("energy", 0.5), 0.0), 1.0)
    energy_points = 2.0 * (1 - abs(song["energy"] - target_energy))
    score += energy_points
    reasons.append(f"energy {song['energy']:.2f} vs target {target_energy:.2f} (+{energy_points:.2f})")

    target_valence = min(max(user_prefs.get("valence", 0.65), 0.0), 1.0)  # default: dataset average, i.e. neutral
    valence_points = 0.5 * (1 - abs(song["valence"] - target_valence))
    score += valence_points
    reasons.append(f"valence {song['valence']:.2f} vs target {target_valence:.2f} (+{valence_points:.2f})")

    # Optional preferences: each one only scores when the user sets it,
    # so a missing preference never quietly favors some songs over others.
    wanted_tags = {tag.strip().lower() for tag in user_prefs.get("mood_tags", [])}
    if wanted_tags:
        shared = [tag for tag in song["mood_tags"].lower().split("|") if tag in wanted_tags]
        tag_points = min(0.25 * len(shared), 0.75)
        if tag_points:
            score += tag_points
            reasons.append(f"mood tags {'/'.join(shared)} (+{tag_points:.2f})")

    if "decade" in user_prefs:
        gap = abs(song["release_decade"] - user_prefs["decade"])
        decade_points = 0.5 if gap == 0 else 0.25 if gap == 10 else 0.0
        if decade_points:
            score += decade_points
            reasons.append(f"from the {song['release_decade']}s (+{decade_points:.2f})")

    if song["language"].lower() == user_prefs.get("language", "").strip().lower():
        score += 0.25
        reasons.append(f"language {song['language']} (+0.25)")

    themes = {theme.strip().lower() for theme in user_prefs.get("themes", [])}
    if song["lyrical_theme"].lower() in themes:
        score += 0.25
        reasons.append(f"theme {song['lyrical_theme']} (+0.25)")

    # A penalty rather than a filter, so explicit songs still rank if nothing else fits
    if song["explicit"] and user_prefs.get("allow_explicit", True) is False:
        score -= 2.0
        reasons.append("explicit lyrics (-2.0)")

    return score, reasons

def max_score(user_prefs: Dict) -> float:
    """Best possible score for this user: 5.0 base plus any optional preferences they set."""
    best = 5.0
    if user_prefs.get("mood_tags"):
        best += min(0.25 * len(user_prefs["mood_tags"]), 0.75)
    if "decade" in user_prefs:
        best += 0.5
    if user_prefs.get("language"):
        best += 0.25
    if user_prefs.get("themes"):
        best += 0.25
    return best

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song and return the top k as (song, score, explanation), highest score first."""
    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:k]
    return [(song, score, ", ".join(reasons)) for song, score, reasons in ranked]
