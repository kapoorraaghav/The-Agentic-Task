from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_ui_question_and_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Lens"
    app.toggle[0].set_value(True).run()
    app.chat_input[0].set_value("Total Sales by City").run()
    assert not app.exception
    answer = app.session_state["answers"][-1][1]
    assert answer["mode"] == "offline"
    assert answer["trace"][-1]["tool"] == "aggregate"
    assert len(answer["trace"][-1]["result"]["values"]) == 3
