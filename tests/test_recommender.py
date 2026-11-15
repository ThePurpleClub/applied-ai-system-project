from src.recommender import Song, UserProfile, Recommender

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""
    assert "genre match" in explanation


def test_recommend_ranks_by_score_not_list_order():
    user = UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.4,
        likes_acoustic=True,
    )
    results = make_small_recommender().recommend(user, k=2)

    assert results[0].genre == "lofi"
    assert results[0].mood == "chill"
    assert results[1].genre == "pop"


def test_derived_mood_matches_without_a_bonus():
    from src.recommender import score_song

    base = {
        "genre": "rock", "mood": "sad", "energy": 0.3, "valence": 0.2,
        "mood_tags": "", "release_decade": 2010, "language": "en",
        "explicit": False, "lyrical_theme": "none",
    }
    prefs = {"mood": "sad", "energy": 0.3, "valence": 0.2}
    labeled, _ = score_song(prefs, dict(base, mood_source="labeled"))
    derived, reasons = score_song(prefs, dict(base, mood_source="derived"))

    assert labeled - derived == 1.0
    assert any("no bonus" in reason for reason in reasons)


def test_unknown_decade_and_language_never_score():
    from src.recommender import score_song

    song = {
        "genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8, "mood_tags": "",
        "release_decade": None, "language": "", "explicit": False, "lyrical_theme": "none",
    }
    _score, reasons = score_song({"energy": 0.8, "decade": 1980}, song)
    assert not any("from the" in reason or "language" in reason for reason in reasons)
