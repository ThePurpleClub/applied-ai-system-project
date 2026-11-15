"""Compare catalog-only parsing with catalog plus listening guides.

This script does not call Claude. Both sides use the offline keyword parser.
The guides side then fills neutral fields from data/knowledge.
"""

from src.knowledge import apply_guide_hints, retrieve_guides
from src.profile_parser import keyword_profile
from src.recommender import load_songs, recommend_songs

QUERY = "songs for a late-night drive"


def _lines(profile, ranked) -> list:
    lines = [
        f"genre={profile.get('genre') or '(none)'} mood={profile.get('mood') or '(none)'} "
        f"energy={profile.get('energy')} valence={profile.get('valence')} "
        f"tags={profile.get('mood_tags') or []}"
    ]
    for index, (song, score, _reasons) in enumerate(ranked, start=1):
        lines.append(f"{index}. {song['title']} by {song['artist']} ({song['genre']}, {song['mood']}, {score:.2f})")
    return lines


def compare(query: str = QUERY, songs=None, k: int = 5) -> str:
    songs = list(songs) if songs is not None else load_songs("data/songs.csv")
    before = keyword_profile(query, songs)
    before_ranked = recommend_songs(before, songs, k=k)
    guides = retrieve_guides(query)
    after = apply_guide_hints(before, guides)
    after_ranked = recommend_songs(after, songs, k=k)
    lines = [
        "# source: offline. No Claude calls.",
        f"Query: {query}",
        "",
        "Before (catalog keywords only)",
        *_lines(before, before_ranked),
        "",
        "After (keywords plus retrieved guides)",
    ]
    for guide in guides:
        lines.append(f"guide: {guide['title']} ({guide['source']}, {guide['score']})")
    lines.extend(_lines(after, after_ranked))
    return "\n".join(lines)


def main() -> int:
    print(compare())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
