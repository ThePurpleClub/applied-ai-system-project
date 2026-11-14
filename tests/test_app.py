from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_fallback_shows_recommendations_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.pipeline.load_local_env", lambda: None)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception

    app.text_input[0].set_value("upbeat songs for a late-night drive").run()
    app.slider[0].set_value(3).run()
    app.button[0].click().run()

    assert not app.exception
    warnings = " ".join(item.value for item in app.warning)
    markdown = " ".join(item.value for item in app.markdown)
    assert "Claude was not used" in warnings
    assert "ANTHROPIC_API_KEY is not set" in warnings
    assert "Sunrise City" in markdown
    assert app.expander
    expander_text = " ".join(item.value for item in app.expander[0].markdown)
    assert "Sunrise City" in expander_text


def test_streamlit_empty_request_does_not_crash():
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.button[0].click().run()

    assert not app.exception
    warnings = " ".join(item.value for item in app.warning)
    assert "Please enter a music request" in warnings


def test_streamlit_shows_a_message_instead_of_a_traceback(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("catalog exploded")

    # AppTest executes app.py, which imports run_pipeline at that moment.
    monkeypatch.setattr("src.pipeline.run_pipeline", explode)
    ui = AppTest.from_file(APP_PATH, default_timeout=30)
    ui.run()
    ui.text_input[0].set_value("calm lofi").run()
    ui.button[0].click().run()

    assert not ui.exception
    errors = " ".join(item.value for item in ui.error)
    assert "unexpected error" in errors
    assert "catalog exploded" not in errors
    assert "Traceback" not in errors
