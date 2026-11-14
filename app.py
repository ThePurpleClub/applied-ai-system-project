"""Streamlit UI for the music recommender.

Run from the project root:

    streamlit run app.py

This page only collects input and displays what src.pipeline returns.
"""

import logging

import streamlit as st

from src.pipeline import RETRIEVAL_K, run_pipeline

logger = logging.getLogger("recommender")

st.set_page_config(page_title="Music Recommender", page_icon="🎵")
st.title("Music Recommender")
st.caption("Describe what you want to hear. The app parses that request, retrieves songs, and explains the matches.")

request = st.text_input(
    "What do you want to listen to?",
    placeholder="upbeat songs for a late-night drive",
)
count = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)


def show_result(result) -> None:
    """Render one pipeline result. The pipeline already decided the content."""
    if result.mode == "invalid_input":
        st.warning(result.message)
        return
    if result.mode == "fallback":
        st.warning(result.message)
    elif result.hallucinations:
        st.info("Dropped songs that were not in the retrieved set: " + ", ".join(result.hallucinations))
    else:
        st.success(result.message)

    if result.recommendations:
        st.subheader("Recommendations")
        for index, rec in enumerate(result.recommendations, start=1):
            st.markdown(f"**{index}. {rec.title}** by {rec.artist}")
            if rec.score is not None:
                st.caption(f"Retrieval score: {rec.score:.2f}")
            st.write(rec.explanation)

    with st.expander("Parsed profile and retrieved songs"):
        if result.profile:
            st.markdown("**Preference profile**")
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


if st.button("Recommend"):
    try:
        result = run_pipeline(
            request,
            retrieval_k=max(RETRIEVAL_K, int(count)),
            final_k=int(count),
        )
    except Exception:
        logger.exception("ui: recommendation failed")
        st.error("The recommender hit an unexpected error. Details are in logs/recommender.log.")
    else:
        show_result(result)
