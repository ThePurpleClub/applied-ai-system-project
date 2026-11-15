import json
import logging

import pytest

from src.pipeline import configure_logging, keyword_profile, run_pipeline, validate_profile
from src.recommender import recommend_songs


class FakeClaude:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append({"system": system, "user": user})
        if not self.responses:
            raise AssertionError("Claude was called more times than the test scripted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ExplodingClaude:
    def complete(self, system, user):
        raise AssertionError("Claude should not be called")


def song(**overrides):
    base = {
        "id": 1,
        "title": "Sunrise City",
        "artist": "Neon Echo",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.82,
        "tempo_bpm": 118,
        "valence": 0.84,
        "danceability": 0.79,
        "acousticness": 0.18,
        "instrumentalness": 0.02,
        "popularity": 88,
        "release_decade": 2020,
        "mood_tags": "upbeat|bright",
        "language": "en",
        "explicit": False,
        "lyrical_theme": "celebration",
    }
    base.update(overrides)
    return base


POP = song()
LOFI = song(
    id=2,
    title="Midnight Coding",
    artist="LoRoom",
    genre="lofi",
    mood="chill",
    energy=0.42,
    valence=0.56,
    mood_tags="mellow|late-night",
    language="none",
    lyrical_theme="none",
)
METAL = song(
    id=3,
    title="Iron Cathedral",
    artist="Black Meridian",
    genre="metal",
    mood="angry",
    energy=0.97,
    valence=0.22,
    mood_tags="aggressive|dark",
    explicit=True,
    lyrical_theme="rebellion",
)
SONGS = [POP, LOFI, METAL]

POP_PROFILE = {
    "genre": "pop",
    "mood": "happy",
    "energy": 0.85,
    "valence": 0.8,
    "mood_tags": [],
    "language": "",
    "themes": [],
    "allow_explicit": True,
}


def profile_json():
    return json.dumps(POP_PROFILE)


def test_validate_profile_clamps_model_output():
    profile = validate_profile({
        "genre": " Pop ",
        "mood": "HAPPY",
        "energy": 1.8,
        "valence": -3,
        "decade": 1984,
        "mood_tags": ["nostalgic", "late-night"],
        "allow_explicit": False,
        "language": " none ",
    })

    assert profile["genre"] == "Pop"
    assert profile["mood"] == "HAPPY"
    assert profile["energy"] == 1.0
    assert profile["valence"] == 0.0
    assert profile["decade"] == 1980
    assert profile["mood_tags"] == ["nostalgic", "late-night"]
    assert profile["allow_explicit"] is False
    assert profile["language"] == "none"


def test_validate_profile_rejects_wrong_types_and_empty_objects():
    with pytest.raises(ValueError):
        validate_profile({"genre": "pop", "energy": "high"})
    with pytest.raises(ValueError):
        validate_profile({"genre": ["pop"], "energy": 0.8})
    with pytest.raises(ValueError):
        validate_profile({"allow_explicit": "false", "genre": "pop"})
    with pytest.raises(ValueError):
        validate_profile({})


def test_empty_profile_uses_keyword_fallback():
    client = FakeClaude(["{}"])
    result = run_pipeline("upbeat songs for a late-night drive", songs=SONGS, client=client)

    assert result.mode == "fallback"
    assert result.profile["energy"] == 0.85
    assert result.profile["mood"] == "happy"
    assert len(client.calls) == 1


def test_keyword_profile_ignores_negated_words_and_reads_without_explicit():
    negated = keyword_profile("not upbeat and not sad", SONGS)
    assert negated["energy"] == 0.5
    assert negated["mood"] == ""

    clean = keyword_profile("happy pop without explicit lyrics", SONGS)
    assert clean["genre"] == "pop"
    assert clean["allow_explicit"] is False


def test_clean_profile_does_not_retrieve_explicit_songs():
    explicit = song(id=9, title="Gym Hero", artist="Max Pulse", energy=0.99, explicit=True)
    quiet = song(
        id=10,
        title="Library Rain",
        artist="Paper Lanterns",
        genre="lofi",
        mood="chill",
        energy=0.3,
        explicit=False,
    )
    profile = dict(POP_PROFILE, allow_explicit=False, energy=0.99)
    client = FakeClaude([
        json.dumps(profile),
        json.dumps({"recommendations": [
            {"title": "Library Rain", "artist": "Paper Lanterns", "explanation": "A calm lofi song."}
        ]}),
    ])
    result = run_pipeline("clean pop", songs=[explicit, quiet], client=client)

    assert "Gym Hero" not in client.calls[1]["user"]
    assert result.recommendations[0].title == "Library Rain"


def test_conflicting_explanation_is_replaced_with_scorer_reasons():
    client = FakeClaude([
        profile_json(),
        json.dumps({"recommendations": [
            {"title": "Sunrise City", "artist": "Neon Echo", "explanation": "This metal song is angry."}
        ]}),
    ])
    result = run_pipeline("happy pop", songs=SONGS, client=client)

    explanation = result.recommendations[0].explanation
    assert "metal" not in explanation.lower()
    assert "energy" in explanation


def test_model_text_is_stripped_of_control_characters():
    client = FakeClaude([
        profile_json(),
        json.dumps({"recommendations": [
            {"title": "Sunrise City", "artist": "Neon Echo", "explanation": "Bright pop.\n\u001b[31minjected"}
        ]}),
    ])
    result = run_pipeline("happy pop", songs=SONGS, client=client)

    explanation = result.recommendations[0].explanation
    assert "\n" not in explanation
    assert "\u001b" not in explanation
    assert "Bright pop." in explanation


def test_unwritable_log_file_still_recommends(monkeypatch):
    recommender_logger = logging.getLogger("recommender")
    recommender_logger.handlers.clear()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("src.pipeline.logging.FileHandler", boom)
    try:
        result = run_pipeline(
            "happy pop",
            songs=SONGS,
            client=FakeClaude([
                profile_json(),
                json.dumps({"recommendations": [
                    {"title": "Sunrise City", "artist": "Neon Echo", "explanation": "Bright pop."}
                ]}),
            ]),
        )
    finally:
        recommender_logger.handlers.clear()

    assert result.mode == "claude"
    assert result.recommendations[0].title == "Sunrise City"


def test_pipeline_sends_only_retrieved_songs_to_claude():
    client = FakeClaude([
        profile_json(),
        json.dumps({"recommendations": [
            {"title": "Sunrise City", "artist": "Neon Echo", "explanation": "Bright pop for a high-energy mood."}
        ]}),
    ])
    result = run_pipeline("happy pop", songs=SONGS, client=client, retrieval_k=2)

    retrieved = recommend_songs(POP_PROFILE, SONGS, k=2)
    retrieved_titles = {item[0]["title"] for item in retrieved}
    generate_prompt = client.calls[1]["user"]

    assert result.mode == "claude"
    assert result.recommendations[0].title == "Sunrise City"
    assert "Bright pop" in result.recommendations[0].explanation
    for title in retrieved_titles:
        assert title in generate_prompt
    left_out = {song["title"] for song in SONGS} - retrieved_titles
    assert left_out
    for title in left_out:
        assert title not in generate_prompt


def test_pipeline_drops_hallucinated_songs_and_keeps_real_ones():
    client = FakeClaude([
        profile_json(),
        json.dumps({"recommendations": [
            {"title": "Not A Real Song", "artist": "Nobody", "explanation": "Invented."},
            {"title": "Sunrise City", "artist": "Neon Echo", "explanation": "This one was retrieved."},
        ]}),
    ])
    result = run_pipeline("happy pop", songs=SONGS, client=client)

    assert result.mode == "claude"
    assert [rec.title for rec in result.recommendations] == ["Sunrise City"]
    assert result.hallucinations == ["Not A Real Song"]
    assert "Ignored titles" in result.message


def test_pipeline_falls_back_when_every_song_is_hallucinated():
    client = FakeClaude([
        profile_json(),
        json.dumps({"recommendations": [
            {"title": "Not A Real Song", "artist": "Nobody", "explanation": "Invented."}
        ]}),
    ])
    result = run_pipeline("happy pop", songs=SONGS, client=client, final_k=1)

    assert result.mode == "fallback"
    assert result.recommendations[0].title == "Sunrise City"
    assert result.hallucinations == ["Not A Real Song"]
    assert "retrieved set" in result.message


def test_pipeline_falls_back_on_malformed_profile_json():
    client = FakeClaude(["this is not json"])
    result = run_pipeline("upbeat songs for a late-night drive", songs=SONGS, client=client)

    assert result.mode == "fallback"
    assert result.profile["energy"] == 0.85
    assert result.profile["mood"] == "happy"
    assert "late-night" in result.profile["mood_tags"]
    assert result.recommendations


def test_api_error_during_generation_falls_back():
    client = FakeClaude([profile_json(), RuntimeError("api down")])
    result = run_pipeline("happy pop", songs=SONGS, client=client, final_k=1)

    assert result.mode == "fallback"
    assert "api down" in result.message
    assert result.recommendations[0].title == "Sunrise City"


def test_pipeline_falls_back_when_generation_is_invalid():
    client = FakeClaude([profile_json(), "nope"])
    result = run_pipeline("happy pop", songs=SONGS, client=client, final_k=1)

    assert result.mode == "fallback"
    assert result.recommendations[0].title == "Sunrise City"
    assert "invalid" in result.message


def test_missing_api_key_uses_keyword_profile_and_scorer(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.pipeline.load_local_env", lambda: None)
    result = run_pipeline("calm instrumental lofi for studying", songs=SONGS, client=None)

    assert result.mode == "fallback"
    assert "ANTHROPIC_API_KEY is not set" in result.message
    assert result.profile["genre"] == "lofi"
    assert result.profile["language"] == "none"
    assert result.recommendations[0].title == "Midnight Coding"


def test_empty_and_overlong_requests_do_not_call_claude():
    empty = run_pipeline("   ", songs=SONGS, client=ExplodingClaude())
    overlong = run_pipeline("a" * 501, songs=SONGS, client=ExplodingClaude())

    assert empty.mode == "invalid_input"
    assert "Please enter" in empty.message
    assert overlong.mode == "invalid_input"
    assert "too long" in overlong.message


def test_steps_are_logged(caplog):
    caplog.set_level(logging.INFO, logger="recommender")
    client = FakeClaude([
        profile_json(),
        json.dumps({"recommendations": [
            {"title": "sunrise city", "artist": "neon echo", "explanation": "Case-insensitive match."}
        ]}),
    ])
    result = run_pipeline("happy pop", songs=SONGS, client=client)
    text = caplog.text

    assert result.recommendations[0].title == "Sunrise City"
    for step in ("parse:", "retrieve:", "generate:", "validate:", "fallback:"):
        assert step in text


def test_configure_logging_writes_a_file(tmp_path, monkeypatch):
    log_path = tmp_path / "recommender.log"
    monkeypatch.setattr("src.pipeline.LOG_PATH", log_path)
    recommender_logger = logging.getLogger("recommender")
    recommender_logger.handlers.clear()

    try:
        configure_logging()
        recommender_logger.info("parse: hello from the test")
        assert "parse: hello from the test" in log_path.read_text(encoding="utf-8")
    finally:
        recommender_logger.handlers.clear()
