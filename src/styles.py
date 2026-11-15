"""Baseline and music-curator renderers.

The offline renderers are templates, so a comparison does not call Claude.
The curator prompt is the few-shot text used when a live key is set and the
listener picks the curator style.
"""

from typing import Dict, List, Sequence

MAX_REASON_WORDS = 12

CURATOR_SYSTEM = """You are a music curator. Follow this format exactly.
Playlist: <2 to 4 word title>
Vibe: <one short line>
1. <title> — <reason in 12 words or fewer>
2. <title> — <reason in 12 words or fewer>

Example:
Playlist: Rain On The Window
Vibe: soft songs for staying in.
1. Library Rain — calm lofi with almost no vocals.
2. Midnight Coding — mellow energy for a quiet night.

Use only songs you were given. Do not invent titles.
"""


def _words(text: str, limit: int = MAX_REASON_WORDS) -> str:
    return " ".join(str(text).split()[:limit])


def baseline_text(query: str, profile: Dict, recommendations: Sequence) -> str:
    """Plain list. This is the baseline the curator is compared with."""
    lines = [f"Request: {query}", "Songs:"]
    if not recommendations:
        lines.append("(none)")
        return "\n".join(lines)
    for index, rec in enumerate(recommendations, start=1):
        lines.append(f"{index}. {rec.title} by {rec.artist}: {rec.explanation}")
    return "\n".join(lines)


def curator_text(query: str, profile: Dict, recommendations: Sequence) -> str:
    """Fixed curator layout: playlist title, one-line vibe, short reasons."""
    mood = str(profile.get("mood") or "mixed").strip()
    genre = str(profile.get("genre") or "mixed").strip()
    title = " ".join(part for part in (mood, genre, "hour") if part).title()
    vibe = " ".join(part for part in (mood, genre, "songs for this request") if part)
    lines = [f"Playlist: {title}", f"Vibe: {vibe}"]
    if not recommendations:
        lines.append("(none)")
        return "\n".join(lines)
    for index, rec in enumerate(recommendations, start=1):
        lines.append(f"{index}. {rec.title} — {_words(rec.explanation)}")
    return "\n".join(lines)


def format_compliance(text: str) -> float:
    """1 when the curator header and dash reasons are all present."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or not lines[0].startswith("Playlist:") or not lines[1].startswith("Vibe:"):
        return 0.0
    reasons = [line for line in lines[2:] if line[:1].isdigit()]
    if not reasons:
        return 0.0
    good = sum(1 for line in reasons if " — " in line)
    return good / len(reasons)


def average_reason_words(text: str) -> float:
    reasons = []
    for line in text.splitlines():
        if " — " in line:
            reasons.append(line.split(" — ", 1)[1])
        elif ": " in line and line[:1].isdigit():
            reasons.append(line.split(": ", 1)[1])
    if not reasons:
        return 0.0
    return sum(len(reason.split()) for reason in reasons) / len(reasons)


def style_marker_rate(text: str) -> float:
    """Share of curator markers. Baseline text should score near zero."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    marked = sum(1 for line in lines if line.startswith("Playlist:") or line.startswith("Vibe:") or " — " in line)
    return marked / len(lines)


def grounding_rate(text: str, titles: Sequence[str]) -> float:
    """Share of rendered song lines whose title was retrieved."""
    known = {title.lower() for title in titles}
    mentioned = []
    for line in text.splitlines():
        if not line[:1].isdigit() or ". " not in line:
            continue
        body = line.split(". ", 1)[1]
        name = body.split(" — ")[0].split(" by ")[0].strip().lower()
        mentioned.append(name)
    if not mentioned:
        return 1.0
    return sum(1 for name in mentioned if name in known) / len(mentioned)
