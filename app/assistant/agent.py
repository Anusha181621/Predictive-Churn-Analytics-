"""The Claude wiring for the "Ask the Data" assistant.

This is the only module in the project that imports ``anthropic``, and it is deliberately thin.
Everything the assistant can actually *do* lives in :mod:`app.assistant.tools`, which is plain
pandas and is tested without a network call. What is here is the loop, the grounding rules, and
the translation of SDK failures into sentences a business user can act on.

**The key is optional.** :func:`assistant_available` reports whether one is configured, and every
caller checks it first. Without a key the dashboard behaves exactly as it always has -- the
platform's guarantee that it runs locally against four CSV files is not weakened by a feature that
turns itself off when unconfigured.

**The model never supplies a number.** The system prompt below is not decoration; it is the
mechanism that makes an answer checkable. Claude decides which questions to ask of the data and how
to word the reply, and the figures come back from tool calls against the same master frame the
pages render. An answer and the dashboard cannot disagree, because they read the same rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
from anthropic import beta_tool

from app.assistant import tools as tool_module
from src.config.settings import get_settings

__all__ = [
    "AssistantReply",
    "ToolCall",
    "assistant_available",
    "answer",
    "build_tools",
    "system_prompt",
    "EXAMPLE_QUESTIONS",
    "NOT_CONFIGURED",
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

#: Ceiling on tool round-trips in a single answer. Generous for real questions — the longest
#: sensible chain is aggregate, rank, then a handful of lookups — and a backstop against a
#: question that sends the model round in circles at the user's expense.
MAX_ITERATIONS = 12


def assistant_available() -> bool:
    """Whether an API key is configured. Callers must check this before :func:`answer`."""
    return bool(get_settings().anthropic_api_key)


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


def build_tools() -> list[Any]:
    """Wrap the plain tool functions as Claude tools.

    ``beta_tool`` builds each schema from the function's signature and docstring, so the
    description Claude reads and the code that runs are the same artefact and cannot drift apart.
    The wrapping happens here rather than in ``tools.py`` so that module stays importable — and
    testable — without the SDK.
    """
    return [beta_tool(function) for function in tool_module.TOOL_FUNCTIONS]


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


def _explain(error: Exception) -> str:
    """Turn an SDK failure into something a business user can act on.

    Most specific first. A single "something went wrong" would leave a reader unable to tell an
    expired key from a dropped connection, which are fixed by completely different people.
    """
    if isinstance(error, anthropic.AuthenticationError):
        return "The configured API key was rejected. Check `ANTHROPIC_API_KEY` in `.env`."
    if isinstance(error, anthropic.PermissionDeniedError):
        return "The configured API key is not permitted to use this model."
    if isinstance(error, anthropic.NotFoundError):
        return "The configured assistant model does not exist. Check `ASSISTANT_MODEL` in `.env`."
    if isinstance(error, anthropic.RateLimitError):
        return "Too many questions at once. Wait a moment and ask again."
    if isinstance(error, anthropic.APIConnectionError):
        return "Could not reach the assistant service. Check the network connection."
    if isinstance(error, anthropic.APIStatusError):
        if error.status_code >= 500:
            return "The assistant service is having trouble. Try again shortly."
        return f"The assistant could not process that request ({error.status_code})."
    return "The assistant hit an unexpected problem and could not answer."


def _messages(history: list[dict[str, str]], question: str) -> list[dict[str, str]]:
    """Prior turns plus the new question.

    Only the user questions and the final answers are replayed — not the tool traffic behind them.
    Each turn re-queries the data it needs, which keeps the context small and, more importantly,
    stops a stale figure from an earlier turn being reused after the artefacts are regenerated.
    """
    return [*history, {"role": "user", "content": question}]


def stream_answer(
    question: str, history: list[dict[str, str]] | None = None
) -> Iterator[ToolCall | AssistantReply]:
    """Answer ``question``, yielding each tool call as it happens and the reply at the end.

    Yielding the tool calls is what lets the page show its working while the answer is still being
    assembled. The final item is always an :class:`AssistantReply`, carrying either the answer or a
    readable error.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        yield AssistantReply(text=NOT_CONFIGURED, error="not_configured")
        return

    calls: list[ToolCall] = []
    text_blocks: list[str] = []

    try:
        runner = _client().beta.messages.tool_runner(
            model=settings.assistant_model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system_prompt(),
            tools=build_tools(),
            max_iterations=MAX_ITERATIONS,
            # The prompt and the six tool schemas are byte-identical on every request, so they
            # are worth caching across the turns of a conversation. Caching needs a prefix of
            # about a thousand tokens to engage and is silently skipped below that -- if this is
            # ever tuned, check `usage.cache_read_input_tokens` rather than assuming it took.
            cache_control={"type": "ephemeral"},
            messages=_messages(list(history or []), question),
        )
        for message in runner:
            for block in message.content:
                if block.type == "tool_use":
                    call = ToolCall(name=block.name, arguments=dict(block.input or {}))
                    calls.append(call)
                    yield call
                elif block.type == "text" and block.text.strip():
                    text_blocks.append(block.text.strip())
    except Exception as error:  # noqa: BLE001 - every failure becomes a readable sentence
        yield AssistantReply(text=_explain(error), tool_calls=calls, error=type(error).__name__)
        return

    # Only the final turn's text is the answer. Earlier turns are narration between tool calls
    # ("let me check the segments"), and replaying them would read as a stream of consciousness.
    yield AssistantReply(text=text_blocks[-1] if text_blocks else "", tool_calls=calls)


def answer(question: str, history: list[dict[str, str]] | None = None) -> AssistantReply:
    """Answer ``question`` and return the reply, discarding the intermediate tool-call events."""
    reply = AssistantReply(text="")
    for event in stream_answer(question, history):
        if isinstance(event, AssistantReply):
            reply = event
    return reply
