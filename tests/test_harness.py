import json

from eval.run_eval import evaluate_cases, format_case_report, main
from src.agent import execute_tool, run_tool_loop
from tests.test_pipeline import SONGS


def test_offline_cases_pass():
    report = evaluate_cases(offline=True)
    text = format_case_report(report)

    assert report["total"] == 6
    assert report["pass_rate"] == 1
    assert "PASS" in text
    assert "Overall pass rate: 6/6" in text
    assert main(["--offline"]) == 0


def test_tool_loop_recommends_only_retrieved_songs():
    class ScriptedTools:
        supports_tools = True

        def __init__(self):
            self.turns = 0

        def complete(self, system, user):
            return json.dumps({
                "genre": "pop",
                "mood": "happy",
                "energy": 0.85,
                "valence": 0.8,
                "mood_tags": [],
                "themes": [],
                "language": "",
                "allow_explicit": True,
            })

        def tool_turn(self, messages, system, tools):
            self.turns += 1
            if self.turns == 1:
                return {
                    "calls": [{"id": "call-1", "name": "search_catalog", "input": {"genre": "pop", "k": 3}}],
                    "text": "",
                    "assistant": "search",
                }
            return {
                "calls": [],
                "text": json.dumps({"recommendations": [
                    {"title": "Not A Real Song", "artist": "Nobody", "explanation": "Invented."},
                    {"title": "Sunrise City", "artist": "Neon Echo", "explanation": "Bright pop."},
                ]}),
                "assistant": "answer",
            }

    from src.pipeline import run_pipeline

    result = run_pipeline("happy pop", songs=SONGS, client=ScriptedTools())

    assert result.mode == "claude"
    assert [rec.title for rec in result.recommendations] == ["Sunrise City"]
    assert "Not A Real Song" in result.hallucinations
    assert any(step.get("name") == "search_catalog" for step in result.agent_trace)


def test_tool_loop_stops_at_the_step_limit():
    class LoopingTools:
        supports_tools = True

        def complete(self, system, user):
            return json.dumps({
                "genre": "pop",
                "mood": "happy",
                "energy": 0.8,
                "valence": 0.8,
                "mood_tags": [],
                "themes": [],
                "language": "",
                "allow_explicit": True,
            })

        def tool_turn(self, messages, system, tools):
            return {
                "calls": [{"id": "again", "name": "search_catalog", "input": {"k": 1}}],
                "text": "",
                "assistant": "again",
            }

    trace, _retrieved, text = run_tool_loop(LoopingTools(), "happy pop", {"genre": "pop", "energy": 0.8, "valence": 0.7, "allow_explicit": True, "mood": "happy", "mood_tags": [], "themes": []}, SONGS, 2, max_steps=3)

    assert text == ""
    assert any(step.get("kind") == "limit" for step in trace)
    assert sum(1 for step in trace if step.get("name") == "search_catalog") == 3


def test_check_constraints_rejects_unknown_titles():
    state = {"songs": SONGS, "profile": {"genre": "pop", "allow_explicit": True}, "retrieved": [], "final_k": 2}
    execute_tool("search_catalog", {"genre": "pop", "k": 2}, state)
    checked = execute_tool("check_constraints", {"titles": ["Not A Real Song"]}, state)

    assert checked["ok"] is False
    assert any("not retrieved" in problem for problem in checked["problems"])
