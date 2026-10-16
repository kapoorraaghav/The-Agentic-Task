import io
import json
from pathlib import Path
import pytest
from agent import Agent, Analysis, Memory, clean

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def orders(tmp_path):
    return Analysis(
        ROOT / "tests/fixtures/orders.csv", Memory(tmp_path / "m.sqlite"), "orders"
    )


def test_supermarket_totals(tmp_path):
    a = Analysis(ROOT / "data/SuperMarket Analysis.csv", Memory(tmp_path / "m.sqlite"))
    assert a.profile()["rows"] == 1000
    assert len(a.df.columns) == 17
    assert a.aggregate("Sales")["values"][0]["value"] == pytest.approx(322966.749)
    assert a.trend("Date", "Sales", "monthly")["invalid_or_missing_dates"] == 0


def test_all_six_tools(orders):
    assert orders.profile()["rows"] == 10
    assert orders.aggregate("revenue")["values"][0]["value"] == 2070
    assert orders.distribution("revenue")["missing"] == 1
    assert orders.correlation("revenue", "units")["paired_rows"] == 9
    t = orders.trend("date", "revenue", "monthly")
    assert t["invalid_or_missing_dates"] == 1
    assert [b["value"] for b in t["buckets"]] == [500, 500, 1000]
    assert orders.outliers("revenue")["rows"] == [{"csv_line": 10, "value": 1000}]


def test_memory_restart_update_delete(orders):
    orders.remember(
        "active customers",
        "filter",
        filters=[{"column": "status", "op": "ne", "value": "cancelled"}],
    )
    a = Analysis(
        ROOT / "tests/fixtures/orders.csv", Memory(orders.memory.path), "orders"
    )
    result = Agent(a, True).ask("Total revenue for active customers")
    assert result["trace"][-1]["result"]["values"][0]["value"] == 1570
    a.remember(
        "active customers",
        "filter",
        filters=[{"column": "region", "op": "eq", "value": "North"}],
    )
    assert (
        a.aggregate("revenue", named_filter="active customers")["values"][0]["value"]
        == 1630
    )
    a.forget("active customers")
    assert a.recall()["facts"] == []


def test_scope_isolation(orders):
    orders.remember("weekly", "preference", "weekly")
    other = Analysis(ROOT / "data/SuperMarket Analysis.csv", Memory(orders.memory.path))
    assert other.recall()["facts"] == []


def test_preference_changes_future_tool(orders):
    orders.remember("time granularity", "preference", "monthly")
    r = Agent(orders, True).ask("Revenue trend by date")
    assert r["trace"][-1]["arguments"]["frequency"] == "monthly"


def test_missing_empty_and_constant(tmp_path):
    a = Analysis(
        io.BytesIO(b"x,y,g\n,2,a\n,2,a\n,2,b\n,2,b\n"), Memory(tmp_path / "m.sqlite")
    )
    assert clean(a.aggregate("x")["values"])[0]["value"] is None
    assert clean(a.correlation("x", "y"))["pearson_r"] is None
    assert (
        a.aggregate(
            operation="count", filters=[{"column": "g", "op": "eq", "value": "z"}]
        )["values"][0]["value"]
        == 0
    )
    assert "error" in a.call("aggregate", {"metric": "g", "operation": "sum"})


def test_no_eval_or_unknown_columns(orders):
    for name, args in [
        ("aggregate", {"metric": '__import__("os")', "operation": "sum"}),
        ("shell", {}),
        ("aggregate", {"metric": "revenue", "operation": "evil"}),
        (
            "aggregate",
            {
                "metric": "revenue",
                "filters": [{"column": "status", "op": "query", "value": "anything"}],
            },
        ),
    ]:
        assert "error" in orders.call(name, args)
    assert (
        "No answer was guessed"
        in Agent(orders, True).ask("Please forecast next year")["answer"]
    )


def test_duplicate_headers(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        Analysis(io.BytesIO(b"x,x\n1,2\n"), Memory(tmp_path / "m.sqlite"))


def test_json_nonfinite_preserves_text():
    assert clean({"label": "NaN Infinity", "value": float("nan")}) == {
        "label": "NaN Infinity",
        "value": None,
    }


def test_evidence_snapshot(orders):
    r = Agent(orders, True).ask("Total revenue")
    orders.call("remember", {"name": "weekly", "kind": "preference", "text": "weekly"})
    assert len(r["trace"]) == 3
    assert r["trace"][-1]["tool"] == "aggregate"


def test_offline_refuses_unparsed_conditions(orders):
    for q in [
        "Total revenue for cancelled customers",
        "Total revenue where units above 3",
        "Count unique region",
        "Total revenue by South",
    ]:
        r = Agent(orders, True).ask(q)
        assert "No answer was guessed" in r["answer"]
        assert len(r["trace"]) == 2


def test_offline_six_analysis_routes(orders):
    questions = {
        "Profile the dataset": "profile",
        "Total revenue": "aggregate",
        "Distribution of revenue": "distribution",
        "Correlation between revenue and units": "correlation",
        "Monthly revenue trend by date": "trend",
        "Outliers in revenue": "outliers",
    }
    for q, tool in questions.items():
        r = Agent(orders, True).ask(q)
        assert r["trace"][-1]["tool"] == tool
        assert "error" not in r["trace"][-1]["result"]


class FakeClient:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.requests = []

    def chat(self, messages):
        self.requests.append(list(messages))
        return next(self.steps)


def tool(tool_name, **arguments):
    return json.dumps({"tool": tool_name, "arguments": arguments})


def test_llm_error_repair_and_evidence(orders):
    fake = FakeClient(
        [
            tool("aggregate", metric="sales", operation="sum"),
            tool("aggregate", metric="revenue", operation="sum"),
            json.dumps({"answer": "Total revenue is 2070 [T4]."}),
        ]
    )
    result = Agent(orders, client=fake).ask("What is total revenue?")
    assert result["answer"] == "Total revenue is 2070 [T4]."
    assert "error" in result["trace"][2]["result"]
    assert result["trace"][3]["result"]["values"][0]["value"] == 2070


def test_llm_uses_persistent_filter(orders):
    orders.remember(
        "active customers",
        "filter",
        filters=[{"column": "status", "op": "ne", "value": "cancelled"}],
    )
    fake = FakeClient(
        [
            tool(
                "aggregate",
                metric="revenue",
                operation="sum",
                named_filter="active customers",
            ),
            json.dumps({"answer": "1570 [T3]"}),
        ]
    )
    result = Agent(orders, client=fake).ask("Total revenue for active customers")
    assert result["trace"][-1]["result"]["values"][0]["value"] == 1570


def test_memory_write_guard(orders):
    fake = FakeClient(
        [
            tool("remember", name="bad", kind="definition", text="bad"),
            json.dumps({"answer": "Cannot save."}),
        ]
    )
    result = Agent(orders, client=fake).ask("Total revenue")
    assert orders.recall()["facts"] == []
    assert "error" in result["trace"][-1]["result"]


def test_bounded_agent_loop(orders):
    fake = FakeClient([tool("profile") for _ in range(10)])
    result = Agent(orders, client=fake).ask("Analyse everything")
    assert "Stopped after 10" in result["answer"]
    assert len(fake.requests) == 10


def test_malformed_model_output_can_recover(orders):
    fake = FakeClient(
        [
            "not json",
            "[]",
            tool("aggregate", metric="revenue", operation="sum"),
            json.dumps({"answer": "2070 [T3]"}),
        ]
    )
    result = Agent(orders, client=fake).ask("Total revenue")
    assert result["trace"][-1]["result"]["values"][0]["value"] == 2070


def test_empty_csv(tmp_path):
    with pytest.raises(ValueError, match="Column names"):
        Analysis(io.BytesIO(b""), Memory(tmp_path / "m.sqlite"))
