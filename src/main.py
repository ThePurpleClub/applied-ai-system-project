"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Taste profile: a late-night studier who wants calm, wordless background music
    user_prefs = {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "valence": 0.60,
        "likes_acoustic": True,
        "prefers_instrumental": True,
        "prefers_popular": False,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 60)
    print("  Top Recommendations")
    print(f"  Profile: {user_prefs['genre']} / {user_prefs['mood']} / energy {user_prefs['energy']:.2f}")
    print("=" * 60)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {song['title']} by {song['artist']}")
        print(f"   Score: {score:.2f} / 5.00   ({song['genre']}, {song['mood']})")
        print("   Why:")
        for reason in explanation.split(", "):
            print(f"     - {reason}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
