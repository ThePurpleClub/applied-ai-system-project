import json

from eval.run_eval import evaluate
from tests.test_pipeline import POP_PROFILE, SONGS, song


class ScriptedClaude:
    def __init__(self, profile, recommendations):
        self.profile = profile
        self.recommendations = recommendations
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.calls % 2 == 1:
            return json.dumps(self.profile)
        return json.dumps({"recommendations": self.recommendations})


def test_evaluate_grounding_consistency_and_fallback():
    client = ScriptedClaude(
        POP_PROFILE,
        [{"title": "Sunrise City", "artist": "Neon Echo", "explanation": "Retrieved pop song."}],
    )
    grounded = evaluate(["happy pop"], repeats=2, client=client, pipeline=_pipeline_with_songs(client))

    assert grounded["fallback_rate"] == 0
    assert grounded["consistency_rate"] == 1
    assert grounded["grounding_rate"] == 1
    assert grounded["details"][0]["titles"][0] == ("Sunrise City",)

    fallback = evaluate(
        ["happy pop"],
        repeats=2,
        pipeline=lambda prompt, client=None: __import__("src.pipeline", fromlist=["run_pipeline"]).run_pipeline(
            prompt,
            songs=SONGS,
            client=_BadGenerate(POP_PROFILE),
        ),
    )
    assert fallback["fallback_rate"] == 1
    assert fallback["consistency_rate"] == 1


class _BadGenerate:
    def __init__(self, profile):
        self.profile = profile
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self.calls % 2 == 1:
            return json.dumps(self.profile)
        return "not json"


def _pipeline_with_songs(client):
    from src.pipeline import run_pipeline

    def _run(prompt, client=client):
        return run_pipeline(prompt, songs=SONGS, client=client)

    return _run


def test_keyword_fallback_profile_reads_explicit_and_decade_hints():
    from src.pipeline import keyword_profile

    songs = SONGS + [song(id=4, title="Night Drive Loop", artist="Neon Echo", genre="synthwave", mood="moody", release_decade=1980)]
    profile = keyword_profile("family-friendly happy pop with no explicit lyrics", songs)
    assert profile["genre"] == "pop"
    assert profile["allow_explicit"] is False

    retro = keyword_profile("moody 1980s synthwave with no lyrics", songs)
    assert retro["genre"] == "synthwave"
    assert retro["decade"] == 1980
    assert retro["language"] == "none"
