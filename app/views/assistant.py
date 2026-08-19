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

#: Set by the question box, consumed on the next run. Streamlit reruns top to bottom, so a box
#: rendered below the thread cannot append to it in the same pass -- it parks the question here.
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

    _new_chat_button(history)

    question = st.session_state.pop(_PENDING_KEY, None)

    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            reply = _answer(question, history)
            st.markdown(reply.text)

        # The tool traffic is deliberately not replayed into the history: each turn re-queries
        # what it needs, so a figure quoted before the artefacts were regenerated can never be
        # reused after.
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply.text})
        st.session_state[_HISTORY_KEY] = history

    # The question box sits below the thread, so it stays where the reader last looked.
    _ask_your_own(bool(history))


def _new_chat_button(history: list[dict[str, str]]) -> None:
    """Drop the current conversation and start again from an empty page.

    Every turn is answered with fresh queries, so nothing is lost but the thread itself. It is
    offered even on an empty page, greyed out, so its position does not move once a chat begins.
    """
    _, right = st.columns([4, 1])
    with right:
        if st.button(
            "Start new chat",
            key="assistant_new_chat",
            width="stretch",
            disabled=not history,
            help="Clear this conversation and begin a new one",
        ):
            st.session_state[_HISTORY_KEY] = []
            st.session_state.pop(_PENDING_KEY, None)
            st.rerun()


def _ask_your_own(started: bool) -> None:
    """The only way in: a box with room to write a question of your own.

    A one-line chat bar suits a follow-up but not the first question, where a reader usually wants
    to say which segment, which period and what they mean by risk. This gives them the space for
    both, and the same box carries the follow-ups once the thread has started.
    """
    if started:
        section("Ask another question")
    else:
        section(
            "Ask your own question",
            "Write it in your own words — the assistant answers from the customer book, the "
            "churn predictions, the segments, the recommended actions and the model's own "
            "accuracy.",
        )
    with st.form("assistant_custom_question", clear_on_submit=True, border=False):
        text = st.text_area(
            "Your question",
            key="assistant_custom_text",
            height=110,
            placeholder=(
                "For example: which of our high-value customers are most likely to churn this "
                "quarter, and what should we do about them?"
            ),
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask", type="primary")

    # Outside the form block: a rerun raised inside it would abandon the form mid-render.
    if asked and text.strip():
        st.session_state[_PENDING_KEY] = text.strip()
        st.rerun()


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


def _unconfigured() -> None:
    """What the page shows when no assistant key has been set up.

    It still explains what the feature would do. A reader who finds an empty page learns nothing;
    a reader who finds this can decide whether it is worth asking for.
    """
    st.info(NOT_CONFIGURED)

    section("What this page does when it is switched on")
    st.markdown(
        "It answers questions of your own about the customer base in plain language, by querying "
        "the same figures the other pages show — the customer book, the churn predictions, the "
        "segments, the recommended actions and the model's own accuracy. It cannot change any of "
        "them, and every figure in an answer comes from a query you can see it make."
    )
    st.markdown(
        "Questions the data cannot support — marketing spend, campaign history, web analytics, "
        "anything about the future beyond the modelled horizon — are declined rather than "
        "answered with an invented figure."
    )
