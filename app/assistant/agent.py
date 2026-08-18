"""The OpenAI wiring for the "Ask the Data" assistant.

This is the only module in the project that imports ``openai``, and it is deliberately thin.
Everything the assistant can actually *do* lives in :mod:`app.assistant.tools`, which is plain
pandas with no provider types in it at all -- that separation is what made swapping the model
provider a change to this one file rather than to the feature.

**The key is optional.** :func:`assistant_available` reports whether one is configured, and every
caller checks it first. Without a key the dashboard behaves exactly as it always has -- the
platform's guarantee that it runs locally against four CSV files is not weakened by a feature that
turns itself off when unconfigured.

**The model never supplies a number.** The system prompt below is not decoration; it is the
mechanism that makes an answer checkable. The model chooses which questions to ask of the data and
how to word the reply, and the figures come back from tool calls against the same master frame the
pages render. An answer and the dashboard cannot disagree, because they read the same rows.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import openai

from app.assistant import tools as tool_module
from src.config.settings import get_settings

__all__ = [
    "AssistantReply",
    "ToolCall",
    "assistant_available",
    "answer",
    "stream_answer",
    "system_prompt",
    "EXAMPLE_QUESTIONS",
    "NOT_CONFIGURED",
    "MAX_ITERATIONS",
]

#: Shown wherever the assistant would be offered but no key is configured. Deliberately free of
#: file paths and variable names: the reader of a dashboard is not necessarily the person who
#: configures it, and the exact setting is documented in the README for whoever is.
NOT_CONFIGURED = (
    "The assistant has not been switched on for this dashboard. Ask whoever set it up to add an "
    "assistant API key to the configuration and restart. Everything else here works without it."
)

#: Seeded on the empty page, chosen to show the range: an aggregate, a ranking that needs
#: chaining, and a question about the model's own trustworthiness.
EXAMPLE_QUESTIONS = (
    "Which customers carry the most revenue at risk, and what should we do about them?",
    "Compare churn risk across our acquisition channels.",
    "How good is the churn model, really?",
)

#: Enough room for a long answer with several tool round-trips behind it.
MAX_TOKENS = 16000

#: Ceiling on tool round-trips in a single answer. Generous for real questions -- the longest
#: sensible chain is aggregate, rank, then a handful of lookups -- and a backstop against a
#: question that sends the model round in circles at the user's expense.
MAX_ITERATIONS = 12


def assistant_available() -> bool:
    """Whether an API key is configured. Callers must check this before :func:`answer`."""
    return bool(get_settings().openai_api_key)


def system_prompt() -> str:
    """The grounding rules.

    These are the substance of the feature rather than politeness. A retention manager acting on a
    figure needs to know it came from the data, that an assumption-dependent number is labelled as
    one, and that a question the data cannot answer gets a refusal instead of a plausible
    invention.
    """
    settings = get_settings()
    return f"""\
You are the analyst for a fashion retailer's customer churn and retention platform. You answer \
questions about the customer base for commercial staff — retention managers, marketers, executives.

HOW YOU ANSWER
Every figure you state must come from a tool call in this conversation. Never state a number from \
memory, and never calculate a new one from numbers you have already quoted — call a tool instead. \
If you cannot get a figure from a tool, say so.

Start with book_summary when you need bearings. Chain tools freely: rank customers, then look one \
up; aggregate a dimension, then drill into the worst group. Call several tools before answering \
when the question needs it.

WHAT THE DATA IS
Churn probability is the modelled likelihood that a customer makes no purchase within \
{settings.churn_inactivity_days} days of the prediction date. It is a probability across a \
population, never a statement about what one person will do. If asked who "will" churn, answer in \
probabilities and say why certainty is not available.

Money is in {settings.currency}.

Revenue at risk is measured: churn probability multiplied by expected future revenue.

Expected retained revenue, campaign ROI and retention propensity all rest on an assumption this \
dataset cannot measure — there is no campaign log and no control group, so the propensity is a \
configured guess. Whenever you quote one of these, say it depends on that assumption. Tool results \
mark them with an ASSUMED suffix. Revenue at risk does not depend on it; do not hedge that one.

WHAT YOU DO NOT HAVE
The platform holds four things: customers, transactions, returns and products. There is no \
marketing spend, no campaign history, no web or app analytics, no competitor data, no customer \
service contacts, and nothing after the prediction date. If a question needs any of that, say \
plainly that you do not have it. Do not estimate it, and do not substitute a proxy without saying \
that is what you are doing.

HOW YOU WRITE
Short and direct. Lead with the answer, then the evidence. Use a short table when comparing \
groups, a list when naming customers. Quote money with the currency and round sensibly — nobody \
needs four decimal places. Name customers by their id. When a number carries a caveat the tools \
reported, pass the caveat on rather than dropping it; the caveats are the honest part."""


@dataclass
class ToolCall:
    """One tool the model called while answering, for the audit trail shown in the UI."""

    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantReply:
    """A finished answer and the tool calls that produced it."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None


def _client() -> openai.OpenAI:
    """The chat client.

    ``base_url`` is passed only when configured, so the default path is plain OpenAI. Set it and
    the same code talks to any OpenAI-compatible endpoint — a gateway such as OpenRouter, Azure, or
    a local server — because only the URL and the model id differ, not the request shape.
    """
    settings = get_settings()
    return openai.OpenAI(
        api_key=settings.openai_api_key,
        **({"base_url": settings.assistant_base_url} if settings.assistant_base_url else {}),
    )


def _explain(error: Exception) -> str:
    """Turn an SDK failure into something a business user can act on.

    Most specific first. A single "something went wrong" would leave a reader unable to tell an
    expired key from a dropped connection, which are fixed by completely different people.
    """
    if isinstance(error, openai.AuthenticationError):
        return "The configured API key was rejected. Check the assistant key in the configuration."
    if isinstance(error, openai.PermissionDeniedError):
        return "The configured API key is not permitted to use this model."
    if isinstance(error, openai.NotFoundError):
        return "The configured assistant model does not exist. Check the model name in `.env`."
    if isinstance(error, openai.RateLimitError):
        return "Too many questions at once, or the account is out of quota. Try again shortly."
    if isinstance(error, openai.APITimeoutError):
        return "The assistant took too long to answer. Try a narrower question."
    if isinstance(error, openai.APIConnectionError):
        return "Could not reach the assistant service. Check the network connection."
    if isinstance(error, openai.APIStatusError):
        if error.status_code >= 500:
            return "The assistant service is having trouble. Try again shortly."
        return f"The assistant could not process that request ({error.status_code})."
    return "The assistant hit an unexpected problem and could not answer."


def _assistant_turn(message: Any) -> dict[str, Any]:
    """The model's own turn, in the shape it has to be replayed as.

    Built explicitly rather than by dumping the response object: the API rejects some of the null
    fields a full dump carries, and a tool call must be echoed back with the same id it arrived
    with or the tool results that follow cannot be matched to it.
    """
    turn: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if getattr(message, "tool_calls", None):
        turn["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ]
    return turn


def _arguments(call: Any) -> dict[str, Any]:
    """Parse a tool call's arguments.

    Always through ``json.loads`` -- the arguments arrive as a JSON *string*, and its escaping is
    the model's business, not ours. Malformed JSON becomes an empty call, which the tool answers
    with a message about what it expected instead of raising.
    """
    try:
        parsed = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def stream_answer(
    question: str, history: list[dict[str, str]] | None = None
) -> Iterator[ToolCall | AssistantReply]:
    """Answer ``question``, yielding each tool call as it happens and the reply at the end.

    Yielding the tool calls is what lets the page show its working while the answer is still being
    assembled. The final item is always an :class:`AssistantReply`, carrying either the answer or a
    readable error.

    Only the prior questions and answers are replayed -- not the tool traffic behind them. Each
    turn re-queries the data it needs, which keeps the conversation small and, more importantly,
    stops a figure quoted before the artefacts were regenerated from being reused after.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        yield AssistantReply(text=NOT_CONFIGURED, error="not_configured")
        return

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        *(history or []),
        {"role": "user", "content": question},
    ]
    calls: list[ToolCall] = []

    try:
        client = _client()
        for _ in range(MAX_ITERATIONS):
            response = client.chat.completions.create(
                model=settings.assistant_model,
                messages=messages,
                tools=tool_module.TOOL_SCHEMAS,
                max_completion_tokens=MAX_TOKENS,
            )
            message = response.choices[0].message
            messages.append(_assistant_turn(message))

            requested = getattr(message, "tool_calls", None)
            if not requested:
                yield AssistantReply(text=(message.content or "").strip(), tool_calls=calls)
                return

            for call in requested:
                arguments = _arguments(call)
                record = ToolCall(name=call.function.name, arguments=arguments)
                calls.append(record)
                yield record
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_module.dispatch(call.function.name, arguments),
                    }
                )
    except Exception as error:  # noqa: BLE001 - every failure becomes a readable sentence
        yield AssistantReply(text=_explain(error), tool_calls=calls, error=type(error).__name__)
        return

    # The loop ran out of turns with the model still calling tools. Saying so is better than
    # presenting whatever half-finished text happens to be last -- a truncated answer that looks
    # complete is worse than no answer.
    yield AssistantReply(
        text=(
            "That question needed more steps than the assistant is allowed to take in one go. "
            "Try asking it in smaller parts."
        ),
        tool_calls=calls,
        error="max_iterations",
    )


def answer(question: str, history: list[dict[str, str]] | None = None) -> AssistantReply:
    """Answer ``question`` and return the reply, discarding the intermediate tool-call events."""
    reply = AssistantReply(text="")
    for event in stream_answer(question, history):
        if isinstance(event, AssistantReply):
            reply = event
    return reply
