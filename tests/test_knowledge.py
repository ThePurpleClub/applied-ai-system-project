from src.knowledge import apply_guide_hints, retrieve_guides
from src.profile_parser import keyword_profile
from src.recommender import load_songs, recommend_songs


def test_late_night_drive_guide_changes_a_vague_profile():
    songs = load_songs("data/songs.csv")
    query = "songs for a late-night drive"
    before = keyword_profile(query, songs)
    guides = retrieve_guides(query)
    after = apply_guide_hints(before, guides)

    assert guides
    assert guides[0]["title"] == "Late-night drive"
    assert before.get("mood") in ("", None)
    assert after["mood"] == "moody"
    assert after["energy"] != before["energy"]
    before_titles = [song["title"] for song, _score, _reasons in recommend_songs(before, songs, k=5)]
    after_titles = [song["title"] for song, _score, _reasons in recommend_songs(after, songs, k=5)]
    assert before_titles != after_titles


def test_explicit_request_is_not_overwritten_by_a_guide():
    songs = load_songs("data/songs.csv")
    profile = keyword_profile("upbeat happy pop", songs)
    updated = apply_guide_hints(profile, retrieve_guides("upbeat happy pop"))

    assert updated["genre"] == "pop"
    assert updated["mood"] == "happy"
    assert updated["energy"] == 0.85
