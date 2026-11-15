"""Follow-up requests, session feedback, and the constraint self-check.

These functions are pure: they read and copy profiles and recommendation
lists, and the pipeline decides what to do with the result.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from src.profile_parser import (
    CLEAN_PHRASES,
    contains_phrase,
    keyword_profile,
    negated_positions,
)

if TYPE_CHECKING:
    from src.pipeline import Recommendation


def copy_profile(profile: Dict) -> Dict:
    copied = dict(profile)
    copied["exclude_artists"] = list(profile.get("exclude_artists") or [])
    copied["exclude_genres"] = list(profile.get("exclude_genres") or [])
    copied["mood_tags"] = list(profile.get("mood_tags") or [])
    copied["themes"] = list(profile.get("themes") or [])
    return copied

def apply_feedback(profile: Dict, feedback: Optional[Sequence[Dict]], songs: Sequence[Dict]) -> Dict:
    """Nudge a profile toward liked songs and away from disliked artists."""
    updated = copy_profile(profile)
    by_pair = {(song["title"].lower(), song["artist"].lower()): song for song in songs}
    for item in feedback or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip().lower()
        artist = str(item.get("artist", "")).strip().lower()
        vote = str(item.get("vote", "")).strip().lower()
        song = by_pair.get((title, artist))
        if song is None:
            continue
        if vote == "down" and song["artist"] not in updated["exclude_artists"]:
            updated["exclude_artists"].append(song["artist"])
        elif vote == "up":
            updated["energy"] = round((float(updated.get("energy", 0.5)) + float(song["energy"])) / 2, 2)
            updated["valence"] = round((float(updated.get("valence", 0.65)) + float(song["valence"])) / 2, 2)
            if not updated.get("mood"):
                updated["mood"] = song["mood"]
            if not updated.get("genre"):
                updated["genre"] = song["genre"]
    return updated

def refine_profile(
    previous: Dict,
    text: str,
    songs: Sequence[Dict],
    feedback: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Update a saved profile from a follow-up sentence and session feedback."""
    profile = copy_profile(previous)
    lowered = text.lower()
    if contains_phrase(lowered, "more chill") or contains_phrase(lowered, "chillier"):
        profile["energy"] = round(max(0.0, float(profile.get("energy", 0.5)) - 0.25), 2)
        profile["valence"] = round(min(float(profile.get("valence", 0.65)), 0.55), 2)
        profile["mood"] = "chill"
    elif contains_phrase(lowered, "more upbeat") or contains_phrase(lowered, "more energy"):
        profile["energy"] = round(min(1.0, float(profile.get("energy", 0.5)) + 0.25), 2)
        profile["valence"] = round(max(float(profile.get("valence", 0.65)), 0.75), 2)
        profile["mood"] = "happy"

    genres = sorted({song["genre"].strip() for song in songs if song.get("genre")}, key=len, reverse=True)
    for genre in genres:
        if negated_positions(lowered, genre.lower()):
            if genre not in profile["exclude_genres"]:
                profile["exclude_genres"].append(genre)
            if str(profile.get("genre", "")).lower() == genre.lower():
                profile["genre"] = ""
        elif contains_phrase(lowered, genre.lower()):
            profile["genre"] = genre

    fresh = keyword_profile(text, songs)
    if fresh.get("mood") and contains_phrase(lowered, fresh["mood"]):
        profile["mood"] = fresh["mood"]
    if fresh.get("allow_explicit") is False:
        profile["allow_explicit"] = False
    if fresh.get("language") == "none":
        profile["language"] = "none"
    if "decade" in fresh:
        profile["decade"] = fresh["decade"]
    return apply_feedback(profile, feedback, songs)


def wants_unique_artists(query: str) -> bool:
    text = query.lower()
    return any(
        contains_phrase(text, phrase)
        for phrase in ("different artists", "no duplicates", "diverse", "variety", "more variety")
    )


def _song_for(rec: "Recommendation", songs: Sequence[Dict]) -> Optional[Dict]:
    for song in songs:
        if song["title"].lower() == rec.title.lower() and song["artist"].lower() == rec.artist.lower():
            return song
    return None


def constraint_issues(query: str, profile: Dict, recommendations: Sequence["Recommendation"], songs: Sequence[Dict]) -> List[str]:
    """Return the constraint failures for one recommendation list."""
    issues = []
    clean = profile.get("allow_explicit", True) is False or any(
        contains_phrase(query.lower(), phrase) for phrase in CLEAN_PHRASES
    )
    if clean:
        explicit_titles = []
        for rec in recommendations:
            song = _song_for(rec, songs)
            if (song and song.get("explicit")) or rec.explicit:
                explicit_titles.append(rec.title)
        if explicit_titles:
            issues.append("explicit songs: " + ", ".join(explicit_titles))
    if wants_unique_artists(query):
        artists = [rec.artist.lower() for rec in recommendations]
        if len(artists) != len(set(artists)):
            issues.append("duplicate artists")
    if profile.get("mood") and recommendations:
        wanted = str(profile["mood"]).lower()
        if not any(rec.mood.lower() == wanted for rec in recommendations):
            issues.append(f"no song matches mood {profile['mood']}")
    if profile.get("energy") is not None and recommendations:
        energies = [rec.energy for rec in recommendations if rec.energy is not None]
        if energies and min(abs(value - float(profile["energy"])) for value in energies) > 0.5:
            issues.append("energy is far from the request")
    return issues
