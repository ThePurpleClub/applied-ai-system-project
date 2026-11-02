"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs, max_score


# Taste profiles to test the recommender against
PROFILES = {
    # A workout listener who wants upbeat, danceable hits
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.85,
        "valence": 0.80,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "prefers_popular": True,
    },
    # A late-night studier who wants calm, wordless background music
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "valence": 0.60,
        "likes_acoustic": True,
        "prefers_instrumental": True,
        "prefers_popular": False,
    },
    # A heavy-music fan who wants loud, aggressive, darker tracks
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "valence": 0.35,
        "likes_acoustic": False,
        "prefers_instrumental": False,
        "prefers_popular": None,
    },
    # --- Adversarial / edge-case profiles ---
    # Conflict: sad mood but very high energy
    "Sad Banger (conflict)": {"genre": "blues", "mood": "sad", "energy": 0.9, "valence": 0.2},
    # Conflict: genre points pull toward slow classical, everything else toward EDM
    "Genre vs Vibe (conflict)": {"genre": "classical", "mood": "euphoric", "energy": 0.95, "valence": 0.9},
    # Messy input: extra whitespace and odd capitalization
    "Messy Input": {"genre": " Pop ", "mood": "HAPPY", "energy": 0.8},
    # Energy outside the 0-1 range
    "Out of Range Energy": {"genre": "metal", "mood": "angry", "energy": 1.5},
    # Genre that doesn't exist in the dataset
    "Unknown Genre": {"genre": "k-pop", "mood": "happy", "energy": 0.8},
    # Nothing matches, all numbers neutral
    "No Signal": {"genre": "none", "mood": "none", "energy": 0.5, "valence": 0.5},
    # --- Profiles using the detailed attributes ---
    # Late-night listener chasing an '80s feel, wants wordless tracks
    "Retro Night Drive": {
        "genre": "synthwave", "mood": "moody", "energy": 0.65, "valence": 0.45,
        "mood_tags": ["nostalgic", "late-night"], "decade": 1980, "language": "none",
    },
    # Same as Sad Banger's mood, but tags let "melancholic" songs count as sad too
    "Melancholy Tags": {
        "genre": "blues", "mood": "sad", "energy": 0.35, "valence": 0.2,
        "mood_tags": ["melancholic", "lonely", "somber"], "themes": ["heartbreak"],
    },
    # Upbeat pop for a car full of kids: no explicit lyrics
    "Family Road Trip": {
        "genre": "pop", "mood": "happy", "energy": 0.85, "valence": 0.8,
        "mood_tags": ["bright", "pumped"], "themes": ["celebration", "motivation"],
        "allow_explicit": False,
    },
}


def print_recommendations(name: str, user_prefs: dict, recommendations: list) -> None:
    """Print one profile's ranked recommendations with scores and reasons."""
    print("\n" + "=" * 60)
    print(f"  Top Recommendations: {name}")
    print(f"  Profile: {user_prefs['genre']} / {user_prefs['mood']} / energy {user_prefs['energy']:.2f}")
    print("=" * 60)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {song['title']} by {song['artist']}")
        print(f"   Score: {score:.2f} / {max_score(user_prefs):.2f}   ({song['genre']}, {song['mood']})")
        print("   Why:")
        for reason in explanation.split(", "):
            print(f"     - {reason}")

    print("\n" + "=" * 60)


def main() -> None:
    songs = load_songs("data/songs.csv")

    for name, user_prefs in PROFILES.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)
        print_recommendations(name, user_prefs, recommendations)


if __name__ == "__main__":
    main()
