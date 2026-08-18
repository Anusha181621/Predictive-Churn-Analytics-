"""Tests for the "Ask the Data" assistant.

The assistant's honesty rests on one property: **every figure it reports comes back from a tool
call**, and the tools read the same artefacts the pages render. So the tools are what is tested
here, thoroughly and entirely offline — no API key, no network, no model. If a tool returns the
wrong rows, no amount of good prompting saves the answer; if it returns the right rows, the answer
is checkable against the dashboard.

The agent layer is tested by replacing the HTTP call and keeping everything else real: the loop,
the tool dispatch, the replay of a tool call and its result, the iteration cap and the error
translation are all ours, so all of them are exercised against a scripted conversation. Nothing
here touches the network or spends a key.
"""

from __future__ import annotations

import copy
import inspect
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

    monkeypatch.setenv("OPENAI_API_KEY", "")
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

    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings(refresh=True)

    def explode() -> None:  # pragma: no cover - the point is that it is never called
        raise AssertionError("a client was constructed without a key")

    monkeypatch.setattr(agent, "_client", explode)
    assert agent.answer("anything at all").error == "not_configured"


def test_every_tool_has_exactly_one_declaration() -> None:
    """A callable with no declaration is unreachable; a declaration with no callable is a 404.

    The schemas are prompt text -- how the model decides which tool answers a question -- so they
    are written out by hand rather than generated. This is what stops them drifting from the code.
    """
    declared = [schema["function"]["name"] for schema in tools.TOOL_SCHEMAS]
    assert sorted(declared) == sorted(tools.TOOLS_BY_NAME)
    assert len(declared) == len(set(declared)), "a tool is declared twice"


def test_no_declaration_offers_a_parameter_the_function_cannot_accept() -> None:
    """The model can only pass what the schema advertises, so the schema must not over-promise."""
    for schema in tools.TOOL_SCHEMAS:
        function = tools.TOOLS_BY_NAME[schema["function"]["name"]]
        parameters = inspect.signature(function).parameters
        offered = set(schema["function"]["parameters"]["properties"])
        assert offered <= set(parameters), (
            f"{function.__name__} is offered {sorted(offered - set(parameters))}, "
            "which it cannot accept"
        )

        required = set(schema["function"]["parameters"].get("required", []))
        mandatory = {
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        assert mandatory <= required, (
            f"{function.__name__} needs {sorted(mandatory - required)} but does not require it"
        )


def test_every_declaration_describes_itself() -> None:
    """A tool with a thin description gets called for the wrong questions."""
    for schema in tools.TOOL_SCHEMAS:
        description = schema["function"]["description"]
        assert len(description) > 60, f"{schema['function']['name']} is barely described"


@requires_artefacts
def test_dispatch_runs_the_named_tool() -> None:
    assert _parsed(tools.dispatch("aggregate", {"dimension": "risk_level"}))["dimension"] == (
        "risk_level"
    )


def test_dispatch_turns_a_bad_call_into_a_tool_result() -> None:
    """A raised exception abandons the answer. A message lets the model fix its own mistake."""
    unknown = _parsed(tools.dispatch("summon_a_dragon", {}))
    assert "error" in unknown
    assert unknown["available"], "the error does not tell the model what it may call instead"

    wrong_arguments = _parsed(tools.dispatch("aggregate", {"nonsense": 1}))
    assert "error" in wrong_arguments


def test_tool_calls_are_surfaced_and_the_last_turn_is_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive the loop with a scripted conversation instead of a model. No network, no key spent.

    Scope, stated plainly: the stub replaces the HTTP call, not the loop -- the loop under test is
    ours, so this does exercise the real tool dispatch, the real echoing of a tool call back with
    its id, and the real decision about when to stop. Two turns, mirroring a real exchange: the
    model asks for a tool, then answers from the result.
    """
    from app.assistant import agent

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    get_settings(refresh=True)

    stub = _StubClient(
        [
            _Message("Let me check the book.", [_Call("c1", "book_summary", "{}")]),
            _Message("There are 1,000 customers."),
        ]
    )
    monkeypatch.setattr(agent, "_client", lambda: stub)
    events = list(agent.stream_answer("how many customers are there?"))

    calls = [event for event in events if isinstance(event, agent.ToolCall)]
    reply = events[-1]

    assert [call.name for call in calls] == ["book_summary"]
    assert isinstance(reply, agent.AssistantReply)
    assert reply.error is None
    # The narration written before the tool ran is not the answer.
    assert reply.text == "There are 1,000 customers."

    # The request must carry the grounding rules and the whole tool surface, or the answer would be
    # ungrounded however well the loop is wired.
    first = stub.requests[0]
    assert first["messages"][0]["role"] == "system"
    assert "Every figure you state must come from a tool call" in first["messages"][0]["content"]
    assert len(first["tools"]) == len(tools.TOOL_SCHEMAS)

    # The second request has to replay the assistant's tool call and its result, keyed by the same
    # id. Without that the provider rejects the turn, and the model loses what it just learned.
    replayed = stub.requests[1]["messages"]
    assert replayed[-2]["tool_calls"][0]["id"] == "c1"
    assert replayed[-1]["role"] == "tool"
    assert replayed[-1]["tool_call_id"] == "c1"
    assert replayed[-1]["content"], "the tool result was replayed empty"


def test_a_runaway_conversation_stops_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that keeps calling tools must not spend the user's money indefinitely.

    Reporting the cap is the honest outcome: presenting the last half-finished text instead would
    hand back a truncated answer that looks complete.
    """
    from app.assistant import agent

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    get_settings(refresh=True)

    forever = [
        _Message("still looking", [_Call(f"c{index}", "book_summary", "{}")]) for index in range(50)
    ]
    stub = _StubClient(forever)
    monkeypatch.setattr(agent, "_client", lambda: stub)
    reply = agent.answer("go round in circles please")

    assert reply.error == "max_iterations"
    assert len(stub.requests) == agent.MAX_ITERATIONS
    assert "smaller parts" in reply.text


def test_malformed_tool_arguments_do_not_crash_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arguments arrive as a JSON string the model wrote. It is entitled to get that wrong."""
    from app.assistant import agent

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    get_settings(refresh=True)

    stub = _StubClient(
        [
            _Message("", [_Call("c1", "aggregate", "{not valid json")]),
            _Message("I could not read that."),
        ]
    )
    monkeypatch.setattr(agent, "_client", lambda: stub)
    reply = agent.answer("break the parser")

    assert reply.error is None
    assert reply.text == "I could not read that."
    # An unparseable call becomes an empty one, and the tool answers with what it expected instead.
    assert json.loads(stub.requests[1]["messages"][-1]["content"])["error"]


def test_an_sdk_failure_becomes_a_sentence_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.assistant import agent

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    get_settings(refresh=True)

    def failing() -> None:
        raise ConnectionResetError("the socket went away")

    monkeypatch.setattr(agent, "_client", failing)
    reply = agent.answer("anything")

    assert reply.error == "ConnectionResetError"
    assert "Traceback" not in reply.text
    assert reply.text.endswith(".")


def test_a_custom_endpoint_is_only_passed_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same code talks to OpenAI or to any compatible gateway; only the URL differs."""
    from app.assistant import agent

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ASSISTANT_BASE_URL", "")
    get_settings(refresh=True)
    assert get_settings().assistant_base_url is None
    assert str(agent._client().base_url).startswith("https://api.openai.com")

    monkeypatch.setenv("ASSISTANT_BASE_URL", "https://openrouter.ai/api/v1")
    get_settings(refresh=True)
    assert str(agent._client().base_url).startswith("https://openrouter.ai")


# --------------------------------------------------------------------------------------
# the stub: a chat client that replays a scripted conversation
# --------------------------------------------------------------------------------------


class _Call:
    """A tool call, in the shape the chat API returns one."""

    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = type("_Function", (), {"name": name, "arguments": arguments})()


class _Message:
    def __init__(self, content: str, tool_calls: list[_Call] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _StubClient:
    """Stands in for ``openai.OpenAI``, replaying scripted responses.

    It replaces only the HTTP call. The tool loop, the dispatch and the message replay are the real
    implementation, which is the part worth testing. ``requests`` keeps every set of keyword
    arguments sent, so a test can assert what actually went over the wire.
    """

    def __init__(self, messages: list[_Message]) -> None:
        self._messages = list(messages)
        self.requests: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs: object) -> object:
        # Deep-copy the messages: the agent mutates its own list as the loop proceeds, and a shared
        # reference would make every recorded request look identical to the last one.
        self.requests.append({**kwargs, "messages": copy.deepcopy(kwargs["messages"])})
        message = self._messages.pop(0)
        return type(
            "_Response", (), {"choices": [type("_Choice", (), {"message": message})()]}
        )()
