"""Streamlit UI for the music recommender.

Run from the project root, including VS Code's terminal:

    streamlit run app.py

The page collects a request and renders what src.pipeline returns.
"""

import csv
import html
import io
import logging
import os
import re

import plotly.graph_objects as go
import streamlit as st

from eval.run_eval import TEST_PROMPTS, evaluate
from src.pipeline import RETRIEVAL_K, SONGS_PATH, load_local_env, run_pipeline
from src.recommender import load_songs, max_score
from src.song_map import project_profile, project_songs

logger = logging.getLogger("recommender")

EXAMPLES = (
    "upbeat songs for a late-night drive",
    "calm instrumental lofi for studying",
    "family-friendly happy pop with no explicit lyrics",
    "heavy metal by different artists",
)

SPOTIFY_ID = re.compile(r"^[A-Za-z0-9]{22}$")

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
    for (title, artist), vote in st.session_state.feedback.items():
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
    for index, rec in enumerate(result.recommendations):
        with st.container(border=True):
            st.markdown(f"### {rec.title}")
            st.markdown(f"*{rec.artist}*")
            chips = "".join(
                f'<span class="chip">{html.escape(label)}</span>'
                for label in (rec.genre, rec.mood, "explicit" if rec.explicit else "")
                if label
            )
            if chips:
                st.markdown(chips, unsafe_allow_html=True)
            if rec.score is not None and ceiling:
                st.caption(f"Match score {rec.score:.2f} / {ceiling:.2f}")
                st.progress(min(max(rec.score / ceiling, 0.0), 1.0))
            st.write(rec.explanation)
            if rec.spotify_id and SPOTIFY_ID.match(rec.spotify_id):
                st.iframe(
                    f"https://open.spotify.com/embed/track/{rec.spotify_id}?theme=0",
                    height=80,
                )
                st.markdown(f"[Open in Spotify](https://open.spotify.com/track/{rec.spotify_id})")
            vote_key = (rec.title, rec.artist)
            current = st.session_state.feedback.get(vote_key)
            like, dislike, note = st.columns([1, 1, 3])
            if like.button("Like", key=f"up-{index}-{vote_key!r}"):
                st.session_state.feedback[vote_key] = "up"
                st.rerun()
            if dislike.button("Not for me", key=f"down-{index}-{vote_key!r}"):
                st.session_state.feedback[vote_key] = "down"
                st.rerun()
            if current == "up":
                note.caption("Saved: you liked this. The next refinement will lean toward it.")
            elif current == "down":
                note.caption("Saved: not for you. The next refinement will avoid this artist.")

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["title", "artist", "genre", "mood", "score", "explanation", "spotify_url"])
    for rec in result.recommendations:
        url = f"https://open.spotify.com/track/{rec.spotify_id}" if rec.spotify_id else ""
        writer.writerow([rec.title, rec.artist, rec.genre, rec.mood, rec.score, rec.explanation, url])
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


MOOD_COLORS = {
    "happy": "#f7c76b",
    "intense": "#f2836b",
    "moody": "#b69cff",
    "sad": "#6c9cf0",
    "chill": "#5fd4b4",
    "other": "#9ca3af",
}


def _song_key(point) -> tuple:
    return (point["title"].lower(), point["artist"].lower())


def _marker_trace(three_d: bool, points, **kwargs):
    """One Scatter or Scatter3d trace for a list of map points."""
    coords = {"x": [p["x"] for p in points], "y": [p["y"] for p in points]}
    if three_d:
        coords["z"] = [p["z"] for p in points]
        return go.Scatter3d(**coords, **kwargs)
    return go.Scatter(**coords, **kwargs)


def build_song_map(model, result, three_d: bool) -> go.Figure:
    points = model["points"]
    retrieved = {_song_key(song) for song, _score, _reasons in result.retrieved}
    recommended = {(rec.title.lower(), rec.artist.lower()) for rec in result.recommendations}
    retrieved_songs = [song for song, _score, _reasons in result.retrieved]
    taste = project_profile(result.profile, model, reference_songs=retrieved_songs)
    figure = go.Figure()

    # Catalog songs, one trace per mood so the legend can toggle moods. There
    # are too many genres for a readable legend, so genre is in the hover text.
    by_mood = {}
    for point in points:
        mood = (point["mood"] or "").lower()
        by_mood.setdefault(mood if mood in MOOD_COLORS else "other", []).append(point)
    for index, mood in enumerate(mood for mood in MOOD_COLORS if mood in by_mood):
        group = by_mood[mood]
        figure.add_trace(_marker_trace(
            three_d,
            group,
            mode="markers",
            name=mood,
            legendgroup="moods",
            legendgrouptitle_text="Mood" if index == 0 else None,
            text=[f"<b>{p['title']}</b><br>{p['artist']}" for p in group],
            customdata=[[p["genre"], p["mood"]] for p in group],
            hovertemplate="%{text}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>",
            marker=dict(
                size=4 if three_d else 7,
                opacity=0.5 if three_d else 0.55,
                color=MOOD_COLORS[mood],
                line=dict(width=0),
            ),
        ))

    shortlist = [p for p in points if _song_key(p) in retrieved and _song_key(p) not in recommended]
    if shortlist:
        figure.add_trace(_marker_trace(
            three_d,
            shortlist,
            mode="markers",
            name="Retrieved, not picked",
            legendgroup="result",
            legendgrouptitle_text="Your request",
            text=[p["title"] for p in shortlist],
            hovertemplate="%{text}<extra>retrieved</extra>",
            marker=dict(
                size=7 if three_d else 13,
                symbol="circle-open",
                color="#e2e8f0",
                line=dict(width=2, color="#e2e8f0"),
            ),
        ))

    chosen = [p for p in points if _song_key(p) in recommended]
    if taste and chosen:
        # One trace for every connector, split by None gaps, keeps the legend short.
        xs, ys, zs = [], [], []
        for p in chosen:
            xs += [taste[0], p["x"], None]
            ys += [taste[1], p["y"], None]
            zs += [taste[2], p["z"], None]
        line = dict(color="rgba(245, 215, 110, 0.7)", width=3 if three_d else 1.5)
        if three_d:
            figure.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=line, hoverinfo="skip", showlegend=False))
        else:
            figure.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=line, hoverinfo="skip", showlegend=False))
    if chosen:
        figure.add_trace(_marker_trace(
            three_d,
            chosen,
            mode="markers+text",
            name="Recommended",
            legendgroup="result",
            text=[p["title"] for p in chosen],
            textposition="top center",
            textfont=dict(color="#fbcfe8", size=11),
            hovertemplate="<b>%{text}</b><extra>recommended</extra>",
            marker=dict(
                size=8 if three_d else 15,
                symbol="diamond",
                color="#f472b6",
                line=dict(width=1, color="white"),
            ),
        ))
    if taste:
        taste_point = [{"x": taste[0], "y": taste[1], "z": taste[2]}]
        figure.add_trace(_marker_trace(
            three_d,
            taste_point,
            mode="markers+text",
            name="Your taste",
            legendgroup="result",
            text=["Your taste"],
            textposition="bottom center",
            textfont=dict(color="#f5d76e", size=13),
            hovertemplate="Your taste<extra></extra>",
            marker=dict(
                size=11 if three_d else 22,
                symbol="circle" if three_d else "star",
                color="#f5d76e",
                line=dict(width=2, color="white"),
            ),
        ))

    labels = model.get("axis_labels") or ["", "", ""]
    explained = model.get("explained") or [0.0, 0.0, 0.0]
    titles = [f"{labels[i]} ({explained[i]:.0%})" if i < len(explained) else labels[i] for i in range(3)]
    layout = dict(
        template="plotly_dark",
        height=720 if three_d else 640,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(itemsizing="constant", bgcolor="rgba(18,20,28,0.6)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#1b1e2b", bordercolor="#c4b5fd", font=dict(color="#f6f3ff")),
        # Keep the camera and zoom when Streamlit reruns the page.
        uirevision="song-map",
    )
    if three_d:
        axis = dict(
            backgroundcolor="rgba(27,30,43,0.6)",
            gridcolor="rgba(196,181,253,0.12)",
            zerolinecolor="rgba(196,181,253,0.3)",
            showspikes=False,
            tickfont=dict(size=9, color="#9ca3af"),
            title_font=dict(size=12, color="#ddd6fe"),
        )
        layout["scene"] = dict(
            xaxis=dict(axis, title=titles[0]),
            yaxis=dict(axis, title=titles[1]),
            zaxis=dict(axis, title=titles[2]),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.85)),
        )
    else:
        layout["xaxis"] = dict(title=titles[0], gridcolor="rgba(196,181,253,0.1)", zeroline=False)
        layout["yaxis"] = dict(title=titles[1], gridcolor="rgba(196,181,253,0.1)", zeroline=False)
    figure.update_layout(**layout)
    return figure


def show_song_map(result) -> None:
    st.subheader("Song Map")
    try:
        stamp = (SONGS_PATH.stat().st_mtime_ns, SONGS_PATH.stat().st_size)
        model = cached_song_map(stamp)
    except Exception:
        logger.exception("ui: song map failed")
        st.error("The song map could not be drawn. Details are in logs/recommender.log.")
        return
    if not model.get("points"):
        st.write("Not enough songs to draw a map.")
        return

    view = st.segmented_control("View", ["3D", "2D"], default="3D", key="map_view") or "3D"
    three_d = view == "3D"
    explained = sum((model.get("explained") or [])[: 3 if three_d else 2])
    st.caption(
        f"Each dot is a catalog song, colored by mood and placed by PCA on energy, valence, danceability, acousticness, "
        f"instrumentalness, and tempo. Each axis is named after the features that move it most. "
        f"This view keeps {explained:.0%} of the variation in those six features. "
        "Your taste takes its energy and valence from your request, and the other four features from "
        "the average of the retrieved songs, because the profile does not store them. Distance here is "
        "not the match score; the score also counts genre, mood, and tags."
        + (" Drag to rotate, scroll to zoom, and click a mood in the legend to hide it." if three_d else "")
    )
    st.plotly_chart(build_song_map(model, result, three_d), width="stretch")


def show_reliability() -> None:
    st.subheader("Reliability")
    st.caption("Runs the same fixed prompts as eval/run_eval.py. With no API key, every run uses the deterministic fallback.")
    load_local_env()
    calls = len(TEST_PROMPTS) * 2 * 2  # prompts x repeats x (parse + generate)
    confirmed = True
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        st.warning(
            f"An API key is set, so this sends about {calls} Claude requests "
            f"({len(TEST_PROMPTS)} prompts, twice each, two calls per run). That costs money and takes a while."
        )
        confirmed = st.checkbox(f"I understand this makes about {calls} paid API calls")
    if st.button("Run evaluation", disabled=not confirmed):
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
    st.toggle(
        "Hide explicit songs",
        key="hide_explicit",
        help=(
            "Removes songs that Spotify marks explicit, plus the starter songs labeled explicit."
        ),
    )
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
