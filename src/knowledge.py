"""Retrieve short listening guides and use them to fill a vague profile.

Guides live in data/knowledge/*.md. Retrieval is TF-IDF, so it runs offline.
Hints are applied only to fields the parser left empty or neutral, so an
explicit request such as "happy pop" is not overwritten.
"""

import re
from pathlib import Path
from typing import Dict, List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.refinement import copy_profile

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge"
_CHUNKS: List[Dict] = []
_VECTORIZER: TfidfVectorizer = None
_MATRIX = None


def load_chunks() -> List[Dict]:
    """Split each guide into one chunk per heading."""
    chunks = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        parts = re.split(r"\n(?=## )", path.read_text(encoding="utf-8"))
        for part in parts:
            lines = [line for line in part.strip().splitlines() if line.strip()]
            if not lines:
                continue
            title = lines[0].lstrip("# ").strip()
            text = "\n".join(lines[1:]).strip()
            if not text:
                continue
            chunks.append({"source": path.name, "title": title, "text": text})
    return chunks


def _fit() -> None:
    global _CHUNKS, _VECTORIZER, _MATRIX
    if _MATRIX is not None:
        return
    _CHUNKS = load_chunks()
    _VECTORIZER = TfidfVectorizer(stop_words="english")
    corpus = [f"{chunk['title']}\n{chunk['text']}" for chunk in _CHUNKS]
    _MATRIX = _VECTORIZER.fit_transform(corpus) if corpus else None


def retrieve_guides(query: str, k: int = 3) -> List[Dict]:
    """Return the closest guide chunks for a request."""
    _fit()
    if not query.strip() or _MATRIX is None or not _CHUNKS:
        return []
    scores = cosine_similarity(_VECTORIZER.transform([query]), _MATRIX).ravel()
    order = scores.argsort()[::-1][:k]
    hits = []
    for index in order:
        if scores[index] <= 0:
            continue
        hits.append({**_CHUNKS[int(index)], "score": round(float(scores[index]), 4)})
    return hits


def hints_from_chunk(text: str) -> Dict[str, str]:
    hints = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("profile:"):
            continue
        for piece in stripped.split(":", 1)[1].split(";"):
            if "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            hints[key.strip()] = value.strip()
    return hints


def apply_guide_hints(profile: Dict, chunks: Sequence[Dict]) -> Dict:
    """Fill neutral profile fields from the best matching guides."""
    updated = copy_profile(profile)
    for chunk in chunks:
        hints = hints_from_chunk(chunk.get("text") or "")
        if hints.get("mood") and not updated.get("mood"):
            updated["mood"] = hints["mood"]
        if hints.get("genre") and not updated.get("genre"):
            updated["genre"] = hints["genre"]
        if hints.get("language") and not updated.get("language"):
            updated["language"] = hints["language"]
        if hints.get("energy") and abs(float(updated.get("energy", 0.5)) - 0.5) < 1e-9:
            updated["energy"] = float(hints["energy"])
        if hints.get("valence") and abs(float(updated.get("valence", 0.65)) - 0.65) < 1e-9:
            updated["valence"] = float(hints["valence"])
        if hints.get("mood_tags") and not updated.get("mood_tags"):
            updated["mood_tags"] = [tag.strip() for tag in hints["mood_tags"].split("|") if tag.strip()]
        if hints.get("allow_explicit", "").lower() == "false":
            updated["allow_explicit"] = False
    return updated


def guide_sentence(chunks: Sequence[Dict]) -> str:
    """One grounded clause naming the guide that informed the profile."""
    if not chunks:
        return ""
    top = chunks[0]
    return f"Guide: {top['title']}."
