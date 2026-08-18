"""Tests for the "Ask the Data" assistant.

The assistant's honesty rests on one property: **every figure it reports comes back from a tool
call**, and the tools read the same artefacts the pages render. So the tools are what is tested
here, thoroughly and entirely offline — no API key, no network, no model. If a tool returns the
wrong rows, no amount of good prompting saves the answer; if it returns the right rows, the answer
is checkable against the dashboard.

The model layer gets two narrower tests: that a scripted tool-use response actually drives the
loop, and that the whole feature turns itself off cleanly when no key is configured. Neither
touches the network.
"""

from __future__ import annotations

import json

import pytest

from src.config.settings import get_settings

pytest.importorskip("streamlit", reason="the assistant is part of the dashboard")

from app import data_access  # noqa: E402
from app.assistant import tools  # noqa: E402

CORE = ("features", "predictions", "scores", "recommendations", "explanations")

requires_artefacts = pytest.mark.skipif(
    bool(data_access.missing(*CORE)),
    reason="generated artefacts missing; run the pipeline scripts first",
)


def _parsed(payload: str) -> dict:
    """Every tool returns JSON. Parsing here also asserts that it is well-formed."""
    return json.loads(payload)


# ======================================================================================
# the tools return what they claim to
# ======================================================================================


#: Arguments that exercise each tool. Keyed by name so a new tool added without an entry here
#: fails the coverage test below rather than quietly going untested.
_CALLS: dict[str, dict] = {
    "book_summary": {},
    "rank_customers": {"limit": 3},
    "customer_detail": {"customer_id": "CUST0001"},
    "aggregate": {"dimension": "risk_level"},
    "churn_drivers": {"limit": 3},
    "model_summary": {},
}


def test_every_tool_is_covered_by_these_tests() -> None:
    """A tool the model can call but nothing here exercises is a tool nobody has checked."""
    assert {f.__name__ for f in tools.TOOL_FUNCTIONS} == set(_CALLS)


@requires_artefacts
@pytest.mark.parametrize("tool", tools.TOOL_FUNCTIONS, ids=lambda f: f.__name__)
def test_every_tool_returns_parseable_json(tool) -> None:
    """A tool result goes straight into the context window; malformed JSON is unrecoverable."""
    result = _parsed(tool(**_CALLS[tool.__name__]))
    assert isinstance(result, dict)
    assert "error" not in result, f"{tool.__name__} failed on a valid call: {result}"


@requires_artefacts
def test_the_book_summary_matches_the_artefacts_it_came_from() -> None:
    """The assistant and the Executive Overview must not be able to disagree."""
    master = data_access.load_customer_master()
    summary = _parsed(tools.book_summary())

    assert summary["customers"] == len(master)
    assert summary["revenue_at_risk"] == pytest.approx(
        float(master["revenue_at_risk"].sum()), abs=0.01
    )
    assert summary["churn_horizon_days"] == get_settings().churn_inactivity_days
    assert sum(summary["customers_by_risk_level"].values()) == len(master)


@requires_artefacts
def test_the_book_summary_flags_its_assumption_dependent_figures() -> None:
    """ROI and retained revenue rest on a propensity nobody measured; the tool must say so."""
    summary = _parsed(tools.book_summary())
    assert "expected_retained_revenue_ASSUMED" in summary
    assert "campaign_cost_ASSUMED" in summary
    # Revenue at risk is free of that assumption and must not be hedged.
    assert "revenue_at_risk" in summary
    assert "revenue_at_risk_ASSUMED" not in summary


@requires_artefacts
def test_ranking_is_ordered_and_respects_its_limit() -> None:
    result = _parsed(tools.rank_customers(order_by="revenue_at_risk", limit=5))
    values = [c["revenue_at_risk"] for c in result["customers"]]

    assert len(values) == 5
    assert values == sorted(values, reverse=True)
    master = data_access.load_customer_master()
    assert values[0] == pytest.approx(float(master["revenue_at_risk"].max()), abs=0.01)


@requires_artefacts
def test_ranking_caps_a_runaway_limit() -> None:
    """"Show me everyone" is a request for the export, not for fifty pages of chat."""
    result = _parsed(tools.rank_customers(limit=10_000))
    assert result["returned"] == tools.MAX_ROWS


@requires_artefacts
def test_a_filter_actually_narrows_the_population() -> None:
    master = data_access.load_customer_master()
    result = _parsed(tools.rank_customers(risk_level="Critical", limit=5))

    expected = int(master["risk_level"].eq("Critical").sum())
    assert result["matching_customers"] == expected
    assert expected < len(master), "the fixture has no non-Critical customers to narrow away"
    assert all(c["risk_level"] == "Critical" for c in result["customers"])


@requires_artefacts
def test_filters_are_case_insensitive() -> None:
    """Nobody types "Critical" with the right capitalisation, and the model should not have to."""
    lower = _parsed(tools.rank_customers(risk_level="critical"))
    exact = _parsed(tools.rank_customers(risk_level="Critical"))
    assert lower["matching_customers"] == exact["matching_customers"]


@requires_artefacts
def test_an_unmatched_filter_names_the_values_that_would_work() -> None:
    """An empty result reads as "no such customers exist"; a typo must not look like a finding."""
    result = _parsed(tools.rank_customers(country="Atlantis"))
    assert "error" in result
    assert result["available"], "the error does not say what the real countries are"


@requires_artefacts
def test_an_unknown_ordering_is_refused_rather_than_silently_substituted() -> None:
    result = _parsed(tools.rank_customers(order_by="vibes"))
    assert "error" in result
    assert set(result["valid_order_by"]) == set(tools.ORDERINGS)


@requires_artefacts
def test_customer_detail_carries_the_profile_and_the_ranked_drivers() -> None:
    master = data_access.load_customer_master()
    customer_id = str(master.iloc[0]["customer_id"])
    result = _parsed(tools.customer_detail(customer_id))

    assert result["customer"]["customer_id"] == customer_id
    assert result["churn_drivers"], "no churn drivers were returned for a real customer"
    ranks = [d["rank"] for d in result["churn_drivers"]]
    assert ranks == sorted(ranks)
    assert all(d["explanation"] for d in result["churn_drivers"])


@requires_artefacts
def test_an_unknown_customer_is_an_answer_not_an_exception() -> None:
    """A raised exception ends the turn; a message lets the model correct itself and retry."""
    result = _parsed(tools.customer_detail("CUST9999999"))
    assert "error" in result
    assert "hint" in result


@requires_artefacts
def test_customer_detail_never_reports_a_number_as_a_string() -> None:
    """numpy int64 through `default=str` renders as "5" -- a quantity the model reads as a label."""
    master = data_access.load_customer_master()
    result = _parsed(tools.customer_detail(str(master.iloc[0]["customer_id"])))
    profile = result["customer"]

    for field in ("total_orders", "recency_days", "lifetime_revenue", "churn_probability"):
        if field in profile:
            assert isinstance(profile[field], (int, float)), f"{field} came back as text"


@requires_artefacts
def test_aggregate_covers_the_whole_population() -> None:
    master = data_access.load_customer_master()
    result = _parsed(tools.aggregate("risk_level", "customers"))
    assert sum(g["customers"] for g in result["groups"]) == len(master)


@requires_artefacts
def test_aggregate_is_sorted_so_the_worst_group_leads() -> None:
    result = _parsed(tools.aggregate("channel", "mean_churn_probability"))
    values = [g["mean_churn_probability"] for g in result["groups"]]
    assert values == sorted(values, reverse=True)


@requires_artefacts
@pytest.mark.parametrize(
    ("dimension", "measure"),
    [("astrology", "customers"), ("segment", "enthusiasm")],
)
def test_aggregate_refuses_what_it_cannot_compute(dimension: str, measure: str) -> None:
    result = _parsed(tools.aggregate(dimension, measure))
    assert "error" in result


@requires_artefacts
def test_churn_drivers_are_ranked_and_keep_their_measured_direction() -> None:
    result = _parsed(tools.churn_drivers(limit=5))
    ranks = [d["rank"] for d in result["drivers"]]

    assert ranks == sorted(ranks)
    assert len(ranks) <= 5
    assert all(d["direction"] for d in result["drivers"])
    assert "non-monotone" in result["note"], "the note must explain what a mixed direction means"


@requires_artefacts
def test_model_summary_quotes_accuracy_against_its_baseline() -> None:
    """A bare accuracy figure flatters the model at a 66% base rate; the baseline must travel."""
    result = _parsed(tools.model_summary())

    assert result["accuracy"] is not None
    assert result["accuracy_predicting_the_majority_class_always"] is not None
    assert result["accuracy_lift_over_that_baseline"] is not None
    assert result["caveats_recorded_by_the_training_run"], "the run's own caveats were dropped"


# ======================================================================================
# the agent layer
# ======================================================================================


def test_the_assistant_is_off_when_no_key_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.assistant import agent

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings(refresh=True)

    assert agent.assistant_available() is False
    reply = agent.answer("anything at all")
    assert reply.error == "not_configured"
    assert reply.text == agent.NOT_CONFIGURED


def test_an_unconfigured_answer_never_reaches_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check has to happen before the client is built, not inside a try/except."""
    from app.assistant import agent

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings(refresh=True)

    def explode() -> None:  # pragma: no cover - the point is that it is never called
        raise AssertionError("a client was constructed without a key")

    monkeypatch.setattr(agent, "_client", explode)
    assert agent.answer("anything at all").error == "not_configured"


def test_the_tool_schemas_are_built_from_the_functions_themselves() -> None:
    """The description Claude reads and the code that runs must be one artefact, not two."""
    from app.assistant import agent

    built = agent.build_tools()
    assert len(built) == len(tools.TOOL_FUNCTIONS)

    names = {t.to_dict()["name"] for t in built}
    assert names == {f.__name__ for f in tools.TOOL_FUNCTIONS}

    by_name = {t.to_dict()["name"]: t.to_dict() for t in built}
    ranking = by_name["rank_customers"]["input_schema"]["properties"]
    assert {"order_by", "limit", "risk_level", "segment"} <= set(ranking)
    assert by_name["customer_detail"]["input_schema"]["required"] == ["customer_id"]


def test_tool_calls_are_surfaced_and_the_final_turn_is_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the agent with a scripted conversation instead of a model. No network, no key spent.

    Scope, stated plainly: the stub stands in for the SDK, so *it* does not execute the tools —
    the real runner does that, and the tools themselves are tested directly above. What is checked
    here is the wiring around the runner: that a tool call reaches the caller so the page can show
    its working, and that the answer returned is the final turn's text rather than the narration
    the model wrote before the tool ran ("Let me check the book.").
    """
    from app.assistant import agent

    class Block:
        def __init__(self, **fields: object) -> None:
            self.__dict__.update(fields)

    class Message:
        def __init__(self, content: list[Block]) -> None:
            self.content = content

    scripted = [
        Message(
            [
                Block(type="text", text="Let me check the book."),
                Block(type="tool_use", name="book_summary", input={}, id="t1"),
            ]
        ),
        Message([Block(type="text", text="There are 1,000 customers.")]),
    ]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    get_settings(refresh=True)
    stub = _StubClient(scripted)
    monkeypatch.setattr(agent, "_client", lambda: stub)
    events = list(agent.stream_answer("how many customers are there?"))

    # The request must carry the grounding rules and the whole tool surface, or the answer would
    # be ungrounded however well the loop is wired.
    assert "Every figure you state must come from a tool call" in stub.recorded["system"]
    assert len(stub.recorded["tools"]) == len(tools.TOOL_FUNCTIONS)
    assert stub.recorded["max_iterations"] == agent.MAX_ITERATIONS

    calls = [e for e in events if isinstance(e, agent.ToolCall)]
    reply = events[-1]

    assert [c.name for c in calls] == ["book_summary"]
    assert isinstance(reply, agent.AssistantReply)
    assert reply.error is None
    assert reply.text == "There are 1,000 customers."
    assert [c.name for c in reply.tool_calls] == ["book_summary"]


def test_an_sdk_failure_becomes_a_sentence_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.assistant import agent

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    get_settings(refresh=True)

    def failing() -> None:
        raise ConnectionResetError("the socket went away")

    monkeypatch.setattr(agent, "_client", failing)
    reply = agent.answer("anything")

    assert reply.error == "ConnectionResetError"
    assert "Traceback" not in reply.text
    assert reply.text.endswith(".")


class _StubClient:
    """Stands in for ``anthropic.Anthropic``, yielding a scripted conversation.

    ``recorded`` keeps the request keyword arguments so a test can assert what was actually sent.
    """

    def __init__(self, messages: list[object]) -> None:
        self._messages = messages
        self.recorded: dict = {}
        self.beta = self

    def tool_runner(self, **kwargs: object) -> list[object]:
        self.recorded.update(kwargs)
        return self._messages

    @property
    def messages(self) -> "_StubClient":
        return self
