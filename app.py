"""Streamlit UI for the music recommender.

Run from the project root, including VS Code's terminal:

    streamlit run app.py

The page collects a request and renders what src.pipeline returns.
"""

import csv
import io
import logging

import plotly.graph_objects as go
import streamlit as st

from eval.run_eval import evaluate
from src.pipeline import RETRIEVAL_K, SONGS_PATH, run_pipeline
from src.recommender import load_songs, max_score
from src.song_map import project_profile, project_songs

logger = logging.getLogger("recommender")

EXAMPLES = (
    "upbeat songs for a late-night drive",
    "calm instrumental lofi for studying",
    "family-friendly happy pop with no explicit lyrics",
    "heavy metal by different artists",
)

FEATURE_LABELS = (
    ("energy", "Energy"),
    ("valence", "Valence"),
    ("danceability", "Danceability"),
    ("acousticness", "Acousticness"),
    ("instrumentalness", "Instrumentalness"),
)

st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="wide")
st.markdown(
    """
    <style>
    .hero {
        padding: 1.5rem 1.7rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #31245f 0%, #171a28 48%, #1a2740 100%);
        border: 1px solid rgba(196, 181, 253, 0.35);
        margin-bottom: 0.8rem;
    }
    .hero h1 { margin: 0 0 0.35rem 0; font-size: 2.15rem; letter-spacing: -0.03em; }
    .hero p { margin: 0; color: #ddd6fe; font-size: 1.02rem; }
    .chip {
        display: inline-block;
        margin: 0 0.35rem 0.35rem 0;
        padding: 0.12rem 0.6rem;
        border-radius: 999px;
        background: rgba(196, 181, 253, 0.16);
        color: #ddd6fe;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "request" not in st.session_state:
    st.session_state.request = ""
if "followup" not in st.session_state:
    st.session_state.followup = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None


def remember_example(prompt: str) -> None:
    st.session_state.request = prompt


def feedback_payload():
    items = []
    for key, vote in st.session_state.feedback.items():
        title, artist = key.split("|", 1)
        items.append({"title": title, "artist": artist, "vote": vote})
    return items


def run_request(text: str, previous=None) -> None:
    hide_explicit = st.session_state.get("hide_explicit", False)
    count = int(st.session_state.get("result_count", 5))
    try:
        st.session_state.result = run_pipeline(
            text,
            retrieval_k=max(RETRIEVAL_K, count),
            final_k=count,
            previous_profile=previous,
            feedback=feedback_payload() if previous else None,
            allow_explicit=False if hide_explicit else None,
        )
    except Exception:
        logger.exception("ui: recommendation failed")
        st.session_state.result = None
        st.error("The recommender hit an unexpected error. Details are in logs/recommender.log.")


def show_notice(result) -> None:
    if result.mode == "invalid_input":
        st.warning(result.message)
        return
    if result.mode == "fallback":
        st.warning(result.message)
    elif result.hallucinations:
        st.info("Dropped songs that were not in the retrieved set: " + ", ".join(result.hallucinations))
    else:
        st.success(result.message)
    if result.self_check:
        check = result.self_check
        if check.get("retried"):
            st.info("Self-check: " + check.get("summary", ""))
        elif not check.get("passed", True):
            st.warning("Self-check: " + check.get("summary", ""))


def show_cards(result) -> None:
    if not result.recommendations:
        return
    ceiling = max_score(result.profile) if result.profile else 5.0
    st.subheader("Your playlist")
    for rec in result.recommendations:
        with st.container(border=True):
            st.markdown(f"### {rec.title}")
            st.markdown(f"*{rec.artist}*")
            chips = "".join(
                f'<span class="chip">{label}</span>'
                for label in (rec.genre, rec.mood, "explicit" if rec.explicit else "")
                if label
            )
            if chips:
                st.markdown(chips, unsafe_allow_html=True)
            if rec.score is not None and ceiling:
                st.caption(f"Match score {rec.score:.2f} / {ceiling:.2f}")
                st.progress(min(max(rec.score / ceiling, 0.0), 1.0))
            st.write(rec.explanation)
            vote_key = f"{rec.title}|{rec.artist}"
            current = st.session_state.feedback.get(vote_key)
            like, dislike, note = st.columns([1, 1, 3])
            if like.button("Like", key=f"up-{vote_key}"):
                st.session_state.feedback[vote_key] = "up"
                st.rerun()
            if dislike.button("Not for me", key=f"down-{vote_key}"):
                st.session_state.feedback[vote_key] = "down"
                st.rerun()
            if current == "up":
                note.caption("Saved: you liked this. The next refinement will lean toward it.")
            elif current == "down":
                note.caption("Saved: not for you. The next refinement will avoid this artist.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["title", "artist", "genre", "mood", "score", "explanation"])
    for rec in result.recommendations:
        writer.writerow([rec.title, rec.artist, rec.genre, rec.mood, rec.score, rec.explanation])
    st.download_button(
        "Download playlist CSV",
        buffer.getvalue(),
        file_name="playlist.csv",
        mime="text/csv",
    )


def show_match_chart(result) -> None:
    st.subheader("Why these songs matched")
    st.caption("Energy and valence are compared with your parsed targets. The other bars are the song's own audio features from the catalog.")
    if not result.recommendations:
        st.write("Recommend something to see the comparison.")
        return
    labels = [label for _key, label in FEATURE_LABELS]
    figure = go.Figure()
    if result.profile:
        figure.add_bar(
            name="Your target",
            x=labels,
            y=[
                result.profile.get("energy"),
                result.profile.get("valence"),
                None,
                None,
                None,
            ],
        )
    for rec in result.recommendations:
        figure.add_bar(
            name=rec.title,
            x=labels,
            y=[getattr(rec, key) for key, _label in FEATURE_LABELS],
        )
    figure.update_layout(
        barmode="group",
        template="plotly_dark",
        yaxis_range=[0, 1],
        height=420,
        legend_title_text="",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")


def show_transparency(result) -> None:
    st.subheader("How the AI decided")
    if result.profile:
        st.markdown("**Parsed preference profile**")
        st.json(result.profile)
    else:
        st.write("No preference profile was built.")
    st.markdown("**Retrieved candidates**")
    if not result.retrieved:
        st.write("No songs were retrieved.")
    for song, score, reasons in result.retrieved:
        st.markdown(
            f"- **{song['title']}** by {song['artist']} "
            f"({song['genre']}, {song['mood']}, score {score:.2f})"
        )
        st.caption(reasons)
    st.markdown("**Dropped or ungrounded songs**")
    if result.hallucinations:
        st.write(", ".join(result.hallucinations))
    else:
        st.write("None.")
    if result.mode == "fallback":
        st.markdown("**Fallback reason**")
        st.warning(result.message)
    if result.self_check:
        st.markdown("**Self-check**")
        st.write(result.self_check.get("summary", ""))
        if result.self_check.get("issues"):
            st.write("First-pass issues: " + "; ".join(result.self_check["issues"]))


@st.cache_data(show_spinner=False)
def cached_song_map(signature: tuple):
    """Cache the PCA fit. The signature changes when the catalog changes."""
    del signature
    return project_songs(load_songs(str(SONGS_PATH)))


def show_song_map(result) -> None:
    st.subheader("Song Map")
    st.caption(
        "Each point is a catalog song, placed by PCA on energy, valence, danceability, "
        "acousticness, instrumentalness, and tempo. The star is your parsed taste. "
        "Danceability, acousticness, instrumentalness, and tempo for that star use the catalog average, "
        "because the profile only stores energy and valence."
    )
    try:
        stamp = (SONGS_PATH.stat().st_mtime_ns, SONGS_PATH.stat().st_size)
        model = cached_song_map(stamp)
    except Exception:
        logger.exception("ui: song map failed")
        st.error("The song map could not be drawn. Details are in logs/recommender.log.")
        return
    points = model.get("points") or []
    if not points:
        st.write("Not enough songs to draw a map.")
        return
    retrieved = {
        (song["title"].lower(), song["artist"].lower())
        for song, _score, _reasons in result.retrieved
    }
    recommended = {(rec.title.lower(), rec.artist.lower()) for rec in result.recommendations}
    profile_xy = project_profile(result.profile, model)
    figure = go.Figure()
    by_genre = {}
    for point in points:
        by_genre.setdefault(point["genre"] or "unknown", []).append(point)
    for genre, group in sorted(by_genre.items()):
        figure.add_trace(go.Scatter(
            x=[point["x"] for point in group],
            y=[point["y"] for point in group],
            mode="markers",
            name=genre,
            text=[f"{point['title']} — {point['artist']}" for point in group],
            customdata=[[point["genre"], point["mood"]] for point in group],
            hovertemplate="%{text}<br>genre %{customdata[0]}<br>mood %{customdata[1]}<extra></extra>",
            marker=dict(size=8, opacity=0.55),
        ))
    retrieved_points = [
        point for point in points
        if (point["title"].lower(), point["artist"].lower()) in retrieved
    ]
    if retrieved_points:
        figure.add_trace(go.Scatter(
            x=[point["x"] for point in retrieved_points],
            y=[point["y"] for point in retrieved_points],
            mode="markers",
            name="Retrieved",
            hoverinfo="skip",
            marker=dict(size=14, symbol="circle-open", color="#f8fafc", line=dict(width=2, color="#f8fafc")),
        ))
    chosen = [
        point for point in points
        if (point["title"].lower(), point["artist"].lower()) in recommended
    ]
    if profile_xy and chosen:
        for point in chosen:
            figure.add_trace(go.Scatter(
                x=[profile_xy[0], point["x"]],
                y=[profile_xy[1], point["y"]],
                mode="lines",
                hoverinfo="skip",
                showlegend=False,
                line=dict(color="rgba(245, 215, 110, 0.55)", width=1),
            ))
    if chosen:
        figure.add_trace(go.Scatter(
            x=[point["x"] for point in chosen],
            y=[point["y"] for point in chosen],
            mode="markers",
            name="Recommended",
            text=[point["title"] for point in chosen],
            hovertemplate="%{text}<extra>recommended</extra>",
            marker=dict(size=16, symbol="diamond", color="#f472b6", line=dict(width=1, color="white")),
        ))
    if profile_xy:
        figure.add_trace(go.Scatter(
            x=[profile_xy[0]],
            y=[profile_xy[1]],
            mode="markers",
            name="Your taste",
            hovertemplate="Your taste<extra></extra>",
            marker=dict(size=20, symbol="star", color="#f5d76e", line=dict(width=1, color="white")),
        ))
    figure.update_layout(
        template="plotly_dark",
        height=640,
        legend_title_text="Genre",
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    )
    st.plotly_chart(figure, width="stretch")


def show_reliability() -> None:
    st.subheader("Reliability")
    st.caption("Runs the same fixed prompts as eval/run_eval.py. With no API key, every run uses the deterministic fallback.")
    if st.button("Run evaluation"):
        try:
            with st.spinner("Running the prompt set twice..."):
                st.session_state.eval_report = evaluate(repeats=2)
        except Exception:
            logger.exception("ui: evaluation failed")
            st.error("The evaluation failed. Details are in logs/recommender.log.")
    report = st.session_state.eval_report
    if not report:
        return
    self_check = report.get("self_check_pass_rate")
    grounding, consistency, fallback, checked = st.columns(4)
    grounding.metric("Grounding", f"{report['grounding_rate']:.0%}")
    consistency.metric("Consistency", f"{report['consistency_rate']:.0%}")
    fallback.metric("Fallback", f"{report['fallback_rate']:.0%}")
    checked.metric("Self-check", "n/a" if self_check is None else f"{self_check:.0%}")
    st.bar_chart(
        {
            "Grounding": report["grounding_rate"],
            "Consistency": report["consistency_rate"],
            "Fallback": report["fallback_rate"],
            "Self-check": 0 if self_check is None else self_check,
        }
    )
    for detail in report["details"]:
        titles = ", ".join(detail["titles"][0]) if detail["titles"] else "(none)"
        st.write(f"**{detail['prompt']}** — {titles}")


st.markdown(
    """
    <div class="hero">
      <h1>Music Recommender</h1>
      <p>Tell it the mood. It parses your words, retrieves real songs, and explains the match.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    st.slider("Number of recommendations", min_value=1, max_value=10, value=5, key="result_count")
    st.toggle("Hide explicit songs", key="hide_explicit")
    st.markdown("**Model status**")
    status_slot = st.empty()


def paint_status() -> None:
    result = st.session_state.result
    if result is None:
        status_slot.caption("No request yet.")
    elif result.mode == "fallback":
        status_slot.caption("Fallback — deterministic scorer. " + result.message)
    elif result.mode == "claude":
        status_slot.caption("Claude — explanations are limited to retrieved songs.")
    else:
        status_slot.caption(result.message)

st.caption("Try an example")
example_columns = st.columns(len(EXAMPLES))
for column, prompt in zip(example_columns, EXAMPLES):
    column.button(prompt, on_click=remember_example, args=(prompt,), key=f"example-{prompt}")

request = st.text_input(
    "What do you want to listen to?",
    key="request",
    placeholder="upbeat songs for a late-night drive",
)
if st.button("Recommend", type="primary"):
    st.session_state.feedback = {}
    run_request(request)

paint_status()
result = st.session_state.result
if result is not None:
    show_notice(result)
    if result.mode != "invalid_input":
        playlist, matches, song_map, decided, reliability = st.tabs(
            ["Playlist", "Why it matched", "Song Map", "How the AI decided", "Reliability"]
        )
        with playlist:
            show_cards(result)
            follow = st.text_input(
                "Refine this list",
                key="followup",
                placeholder="more chill, no pop",
            )
            if st.button("Refine"):
                run_request(follow, previous=result.profile)
                st.rerun()
        with matches:
            show_match_chart(result)
        with song_map:
            show_song_map(result)
        with decided:
            show_transparency(result)
        with reliability:
            show_reliability()
else:
    show_reliability()
