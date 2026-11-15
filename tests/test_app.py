from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def click(app, label):
    matches = [button for button in app.button if button.label == label]
    assert matches, [button.label for button in app.button]
    return matches[0].click().run()


def test_streamlit_fallback_shows_recommendations_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.pipeline.load_local_env", lambda: None)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception

    app.text_input[0].set_value("upbeat songs for a late-night drive").run()
    app.slider[0].set_value(3).run()
    app = click(app, "Recommend")

    assert not app.exception
    warnings = " ".join(item.value for item in app.warning)
    markdown = " ".join(item.value for item in app.markdown)
    assert "Claude was not used" in warnings
    assert "ANTHROPIC_API_KEY is not set" in warnings
    headers = " ".join(getattr(item, "value", "") for item in app.subheader)
    assert "Your playlist" in headers or "Your playlist" in markdown
    captions = " ".join(item.value for item in app.caption)
    assert "Match score" in captions
    assert "Sunrise City" in markdown
    assert any(tab.label == "Song Map" for tab in app.tabs)
    assert any(tab.label == "How the AI decided" for tab in app.tabs)


def test_streamlit_empty_request_does_not_crash():
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app = click(app, "Recommend")

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
    ui = click(ui, "Recommend")

    assert not ui.exception
    errors = " ".join(item.value for item in ui.error)
    assert "unexpected error" in errors
    assert "catalog exploded" not in errors
    assert "Traceback" not in errors


def test_reliability_tab_waits_for_a_request(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("src.pipeline.load_local_env", lambda: None)
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    assert not app.exception
    assert "Run evaluation" not in [button.label for button in app.button]


def test_evaluation_needs_confirmation_when_a_key_is_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("src.pipeline.load_local_env", lambda: None)
    monkeypatch.setattr("src.pipeline.build_client", lambda: (None, "ANTHROPIC_API_KEY is not set"))
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    app.text_input[0].set_value("calm lofi").run()
    app = click(app, "Recommend")

    assert not app.exception
    run = [button for button in app.button if button.label == "Run evaluation"][0]
    assert run.disabled
    assert any("paid API calls" in box.label for box in app.checkbox)
