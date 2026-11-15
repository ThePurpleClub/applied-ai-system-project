"""Turn a listener's sentence or Claude's JSON into a preference profile.

validate_profile checks Claude's parsed JSON. keyword_profile is the rough
reading used when Claude is unavailable. The phrase helpers skip negated
mentions, so "no pop" does not count as asking for pop.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence

# Words that are not catalog labels but still tell the fallback parser what to do.
HIGH_ENERGY_WORDS = ("upbeat", "energetic", "party", "dance", "workout", "gym", "hype")
LOW_ENERGY_WORDS = ("calm", "chill", "relaxing", "relaxed", "study", "studying", "focus", "quiet", "mellow", "soft")
MOOD_SYNONYMS = {
    "happy": ("happy", "upbeat", "cheerful", "fun"),
    "chill": ("chill", "calm", "relaxing", "relaxed", "study", "studying", "mellow"),
    "sad": ("sad", "melancholy", "heartbreak"),
    "intense": ("intense", "aggressive"),
    "moody": ("moody", "brooding"),
    "euphoric": ("euphoric",),
    "focused": ("focus", "focused", "studying"),
}
INSTRUMENTAL_PHRASES = ("instrumental", "wordless", "no lyrics", "without lyrics")
CLEAN_PHRASES = (
    "no explicit",
    "not explicit",
    "non-explicit",
    "without explicit",
    "clean lyrics",
    "family-friendly",
    "family friendly",
    "for kids",
)
_NEGATION = re.compile(r"\b(?:not|no|never|without|isn't|isnt|aren't|arent|don't|dont)\b")


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, including a fenced block or surrounding prose."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def _optional_text(value: Any, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _optional_number(value: Any, field: str, default: float) -> float:
    if value is None:
        return default
    # bool is a subclass of int, and it is not a score.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return min(1.0, max(0.0, float(value)))


def _optional_bool(value: Any, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _optional_text_list(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of strings")
    items = []
    for part in value:
        if not isinstance(part, str):
            raise ValueError(f"{field} must be a list of strings")
        text = part.strip()
        if text:
            items.append(text)
    return items


def neutral_profile() -> Dict:
    return {
        "genre": "",
        "mood": "",
        "energy": 0.5,
        "valence": 0.65,
        "mood_tags": [],
        "themes": [],
        "language": "",
        "allow_explicit": True,
    }


def profile_has_signal(profile: Dict) -> bool:
    """True when the profile says something besides the neutral defaults."""
    if profile.get("genre") or profile.get("mood") or profile.get("language"):
        return True
    if profile.get("mood_tags") or profile.get("themes") or "decade" in profile:
        return True
    if profile.get("allow_explicit") is False:
        return True
    if abs(float(profile.get("energy", 0.5)) - 0.5) > 1e-9:
        return True
    if abs(float(profile.get("valence", 0.65)) - 0.65) > 1e-9:
        return True
    return False


def validate_profile(raw: Any) -> Dict:
    """Clamp a parsed profile, and reject the wrong JSON types or an empty object."""
    if not isinstance(raw, dict):
        raise ValueError("preference profile must be a JSON object")

    profile = {
        "genre": _optional_text(raw.get("genre", ""), "genre"),
        "mood": _optional_text(raw.get("mood", ""), "mood"),
        "energy": _optional_number(raw.get("energy", 0.5), "energy", 0.5),
        "valence": _optional_number(raw.get("valence", 0.65), "valence", 0.65),
        "mood_tags": _optional_text_list(raw.get("mood_tags", []), "mood_tags"),
        "themes": _optional_text_list(raw.get("themes", []), "themes"),
        "language": _optional_text(raw.get("language", ""), "language"),
        "allow_explicit": _optional_bool(raw.get("allow_explicit", True), "allow_explicit", True),
    }

    if "decade" in raw and raw.get("decade") is not None:
        decade = raw.get("decade")
        if isinstance(decade, bool) or not isinstance(decade, (int, float)):
            raise ValueError("decade must be a number or null")
        year = int(decade)
        if 1900 <= year <= 2029:
            profile["decade"] = year - (year % 10)
    if not profile_has_signal(profile):
        raise ValueError("profile has no preference signal")
    return profile


def is_negated(text: str, index: int) -> bool:
    window = text[max(0, index - 16):index]
    return _NEGATION.search(window) is not None


def positive_positions(text: str, phrase: str) -> List[int]:
    positions = []
    for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
        if not is_negated(text, match.start()):
            positions.append(match.start())
    return positions


def contains_phrase(text: str, phrase: str) -> bool:
    return bool(positive_positions(text, phrase))


def _first_phrase_at(text: str, phrases: Sequence[str]) -> Optional[int]:
    positions = [pos for phrase in phrases for pos in positive_positions(text, phrase)]
    return min(positions) if positions else None


def keyword_profile(text: str, songs: Sequence[Dict]) -> Dict:
    """Rough profile used only when Claude cannot parse the request."""
    lowered = text.lower()
    profile = neutral_profile()

    genres = sorted({song["genre"].strip() for song in songs if song.get("genre")}, key=len, reverse=True)
    for genre in genres:
        if contains_phrase(lowered, genre.lower()):
            profile["genre"] = genre
            break

    mood_hits = []
    for mood, phrases in MOOD_SYNONYMS.items():
        at = _first_phrase_at(lowered, phrases)
        if at is not None:
            mood_hits.append((at, mood))
    for song in songs:
        mood = song.get("mood", "").strip()
        positions = positive_positions(lowered, mood.lower()) if mood else []
        if positions:
            mood_hits.append((min(positions), mood.lower()))
    if mood_hits:
        profile["mood"] = min(mood_hits)[1]

    high_at = _first_phrase_at(lowered, HIGH_ENERGY_WORDS)
    low_at = _first_phrase_at(lowered, LOW_ENERGY_WORDS)
    if high_at is not None and (low_at is None or high_at <= low_at):
        profile["energy"] = 0.85
        profile["valence"] = 0.8
    elif low_at is not None:
        profile["energy"] = 0.35
        profile["valence"] = 0.55

    if contains_phrase(lowered, "late-night") or contains_phrase(lowered, "late night"):
        profile["mood_tags"] = ["late-night"]
    if any(contains_phrase(lowered, phrase) for phrase in INSTRUMENTAL_PHRASES):
        profile["language"] = "none"
    if any(contains_phrase(lowered, phrase) for phrase in CLEAN_PHRASES):
        profile["allow_explicit"] = False

    theme_names = sorted(
        {
            song["lyrical_theme"].strip()
            for song in songs
            if song.get("lyrical_theme") and song["lyrical_theme"].strip().lower() != "none"
        },
        key=len,
        reverse=True,
    )
    found_themes = [theme for theme in theme_names if contains_phrase(lowered, theme.lower())]
    if found_themes:
        profile["themes"] = found_themes

    decade_match = re.search(r"\b((?:19|20)\d{2})s?\b", lowered)
    short_decade = re.search(r"\b(\d{2})s\b", lowered)
    if decade_match:
        year = int(decade_match.group(1))
        if 1900 <= year <= 2029:
            profile["decade"] = year - (year % 10)
    elif short_decade:
        yy = int(short_decade.group(1))
        year = 1900 + yy if yy >= 50 else 2000 + yy
        profile["decade"] = year - (year % 10)
    return profile


def negated_positions(text: str, phrase: str) -> List[int]:
    positions = []
    for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
        if is_negated(text, match.start()):
            positions.append(match.start())
    return positions
