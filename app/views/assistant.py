"""Ask the Data -- a Claude analyst that answers questions from the platform's own artefacts.

Every other page answers a question somebody thought to build a page for. This one answers the
rest, by letting a model query the same data the pages render.

The tool calls are shown, not hidden. That is the difference between an assistant a retention
manager can act on and one they have to take on faith: the reader can see that "the Influencer
cohort runs hottest" came from an aggregate over the customer base, and can go and check it on the
Churn Risk page. An answer whose working is invisible is a claim, not an analysis.
"""

from __future__ import annotations

import streamlit as st

from app.assistant.agent import (
    EXAMPLE_QUESTIONS,
    NOT_CONFIGURED,
    AssistantReply,
    ToolCall,
    assistant_available,
    stream_answer,
)
from app.components.layout import page_header, section
from app.data_access import load_customer_master, prediction_date, require

#: Chat history in the shape the API expects, so it can be replayed without translation.
_HISTORY_KEY = "assistant_history"

#: Set by an example button, consumed on the next run. Streamlit reruns top to bottom, so a button
#: pressed below the chat input cannot feed it directly -- it parks the question here instead.
_PENDING_KEY = "assistant_pending"

#: How each tool describes itself while it runs.
_TOOL_LABELS = {
    "book_summary": "Reading the whole customer base",
    "rank_customers": "Ranking customers",
    "customer_detail": "Looking up a customer",
    "aggregate": "Breaking the book down",
    "churn_drivers": "Reading the churn drivers",
    "model_summary": "Checking the model's accuracy",
}


def _describe(call: ToolCall) -> str:
    """A one-line account of a tool call, including the arguments that shaped it."""
    label = _TOOL_LABELS.get(call.name, call.name.replace("_", " ").capitalize())
    arguments = ", ".join(f"{k}={v}" for k, v in call.arguments.items() if v not in ("", None))
    return f"{label} ({arguments})" if arguments else label


def render() -> None:
    require("features", "predictions", "scores", "recommendations", "explanations")

    master = load_customer_master()
    page_header(
        "Ask the Data",
        "Ask a question about the customer base and get an answer drawn from these figures",
        as_of=prediction_date(master),
    )

    if not assistant_available():
        _unconfigured()
        return

    history: list[dict[str, str]] = st.session_state.setdefault(_HISTORY_KEY, [])

    if not history:
        _examples()

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask about churn, revenue at risk, segments or the model")
    pending = st.session_state.pop(_PENDING_KEY, None)
    question = question or pending
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        reply = _answer(question, history)
        st.markdown(reply.text)

    # The tool traffic is deliberately not replayed into the history: each turn re-queries what it
    # needs, so a figure quoted before the artefacts were regenerated can never be reused after.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply.text})
    st.session_state[_HISTORY_KEY] = history


def _answer(question: str, history: list[dict[str, str]]) -> AssistantReply:
    """Run the agent, showing each tool call as it happens."""
    reply = AssistantReply(text="")
    with st.status("Consulting the data...", expanded=True) as status:
        for event in stream_answer(question, history):
            if isinstance(event, ToolCall):
                st.write(_describe(event))
            else:
                reply = event
        consulted = len(reply.tool_calls)
        if reply.error:
            status.update(label="Could not answer", state="error", expanded=True)
        else:
            status.update(
                label=f"Answered from {consulted} data "
                f"{'query' if consulted == 1 else 'queries'}",
                state="complete",
                expanded=False,
            )
    return reply


def _examples() -> None:
    """Three starting questions, chosen to show the range rather than to look impressive."""
    section("Try one of these")
    columns = st.columns(len(EXAMPLE_QUESTIONS), gap="medium")
    for index, (column, example) in enumerate(zip(columns, EXAMPLE_QUESTIONS)):
        with column:
            if st.button(example, key=f"assistant_example_{index}", width="stretch"):
                st.session_state[_PENDING_KEY] = example
                st.rerun()


def _unconfigured() -> None:
    """What the page shows when no assistant key has been set up.

    It still explains what the feature would do. A reader who finds an empty page learns nothing;
    a reader who finds this can decide whether it is worth asking for.
    """
    st.info(NOT_CONFIGURED)

    section("What this page does when it is switched on")
    st.markdown(
        "It answers questions about the customer base in plain language, by querying the same "
        "figures the other pages show — the customer book, the churn predictions, the segments, "
        "the recommended actions and the model's own accuracy. It cannot change any of them, and "
        "every figure in an answer comes from a query you can see it make."
    )
    st.markdown("**Questions it is built to answer:**")
    for example in EXAMPLE_QUESTIONS:
        st.markdown(f"- {example}")
    st.markdown(
        "Questions the data cannot support — marketing spend, campaign history, web analytics, "
        "anything about the future beyond the modelled horizon — are declined rather than "
        "answered with an invented figure."
    )
