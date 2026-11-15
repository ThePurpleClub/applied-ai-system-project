import json

from src.pipeline import Recommendation, apply_self_check, run_pipeline
from tests.test_pipeline import SONGS, song


def _no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.pipeline.load_local_env", lambda: None)


def test_followup_lowers_energy_and_feedback_drops_the_artist(monkeypatch):
    _no_key(monkeypatch)
    previous = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "valence": 0.8,
        "allow_explicit": True,
        "mood_tags": [],
        "themes": [],
    }
    result = run_pipeline(
        "more chill",
        songs=SONGS,
        previous_profile=previous,
        feedback=[{"title": "Sunrise City", "artist": "Neon Echo", "vote": "down"}],
    )

    assert result.mode == "fallback"
    assert result.profile["mood"] == "chill"
    assert result.profile["energy"] < 0.9
    assert "Neon Echo" in result.profile["exclude_artists"]
    assert all(rec.artist != "Neon Echo" for rec in result.recommendations)
    assert result.recommendations[0].title == "Midnight Coding"


def test_claude_refinement_prompt_includes_the_previous_profile():
    class FakeClaude:
        def __init__(self):
            self.calls = []

        def complete(self, system, user):
            self.calls.append(user)
            if len(self.calls) == 1:
                return json.dumps({
                    "genre": "lofi",
                    "mood": "chill",
                    "energy": 0.4,
                    "valence": 0.55,
                    "mood_tags": [],
                    "themes": [],
                    "language": "none",
                    "allow_explicit": True,
                })
            return json.dumps({"recommendations": [
                {"title": "Midnight Coding", "artist": "LoRoom", "explanation": "A chill lofi loop."}
            ]})

    client = FakeClaude()
    previous = {"genre": "pop", "mood": "happy", "energy": 0.9, "valence": 0.8, "allow_explicit": True}
    result = run_pipeline("more chill", songs=SONGS, client=client, previous_profile=previous)

    assert "Previous profile" in client.calls[0]
    assert "pop" in client.calls[0]
    assert result.mode == "claude"
    assert result.profile["mood"] == "chill"
    assert result.recommendations[0].title == "Midnight Coding"


def test_self_check_retries_without_explicit_songs():
    explicit = song(id=9, title="Gym Hero", artist="Max Pulse", energy=0.99, explicit=True)
    clean = song(title="Sunrise City", artist="Neon Echo", energy=0.82, explicit=False)
    profile = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "valence": 0.8,
        "allow_explicit": True,
        "mood_tags": [],
        "themes": [],
    }
    recommendations = [
        Recommendation(
            title="Gym Hero",
            artist="Max Pulse",
            explanation="Loud pop.",
            genre="pop",
            mood="happy",
            energy=0.99,
            explicit=True,
        )
    ]
    new_recs, _retrieved, new_profile, report = apply_self_check(
        "happy pop with no explicit lyrics",
        profile,
        recommendations,
        [],
        [explicit, clean],
        2,
    )

    assert report["retried"] is True
    assert report["passed"] is True
    assert new_profile["allow_explicit"] is False
    assert [rec.title for rec in new_recs] == ["Sunrise City"]


def test_self_check_drops_duplicate_artists_when_variety_is_requested():
    first = song(title="Sunrise City", artist="Neon Echo", genre="pop", mood="happy", energy=0.82)
    second = song(id=8, title="Night Drive Loop", artist="Neon Echo", genre="synthwave", mood="moody", energy=0.75)
    third = song(id=10, title="Rooftop Lights", artist="Indigo Parade", genre="indie pop", mood="happy", energy=0.76)
    profile = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "valence": 0.8,
        "allow_explicit": True,
        "mood_tags": [],
        "themes": [],
    }
    recommendations = [
        Recommendation(title="Sunrise City", artist="Neon Echo", explanation="a", genre="pop", mood="happy", energy=0.82),
        Recommendation(title="Night Drive Loop", artist="Neon Echo", explanation="b", genre="synthwave", mood="moody", energy=0.75),
    ]
    new_recs, _retrieved, _profile, report = apply_self_check(
        "happy pop by different artists",
        profile,
        recommendations,
        [],
        [first, second, third],
        2,
    )

    assert report["retried"] is True
    assert len({rec.artist for rec in new_recs}) == len(new_recs)
    assert "Neon Echo" in {rec.artist for rec in new_recs}
    assert "Indigo Parade" in {rec.artist for rec in new_recs}
