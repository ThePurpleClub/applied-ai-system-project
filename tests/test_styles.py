from src.styles import (
    average_reason_words,
    baseline_text,
    curator_text,
    format_compliance,
    style_marker_rate,
)


class _Rec:
    def __init__(self, title, artist, explanation):
        self.title = title
        self.artist = artist
        self.explanation = explanation


def test_curator_format_differs_from_baseline():
    profile = {"mood": "moody", "genre": "pop"}
    songs = [
            _Rec(
                "Night Drive",
                "Neon Echo",
                "genre match (+1.5), mood match (+1.0), energy 0.75 vs target 0.65 (+1.80), valence 0.49 vs target 0.45 (+0.48), mood tags late-night (+0.25)",
            ),
            _Rec(
                "Library Rain",
                "Paper Lanterns",
                "mood match (+1.0), energy 0.35 vs target 0.35 (+2.00), valence 0.60 vs target 0.55 (+0.47), language none (+0.25)",
            ),
        ]
    baseline = baseline_text("late drive", profile, songs)
    curator = curator_text("late drive", profile, songs)

    assert format_compliance(baseline) == 0
    assert format_compliance(curator) == 1
    assert average_reason_words(curator) <= 12
    assert average_reason_words(curator) < average_reason_words(baseline)
    assert style_marker_rate(curator) > style_marker_rate(baseline)
    assert "Playlist:" in curator
    assert "Vibe:" in curator
