"""
Optional Langfuse tracing. Every function here degrades to a safe no-op
if LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set - tracing is
strictly additive, never a requirement for the pipeline to run, same
philosophy as the LLM provider and web search tool: absence of
credentials degrades gracefully rather than crashing anything.

This file was written against the Langfuse v2 SDK and had to be ported.
Both APIs it used are gone in v3+:

  - `from langfuse.callback import CallbackHandler` - the module no
    longer exists; the LangChain integration moved to
    `langfuse.langchain`, and the handler no longer takes credentials
    itself. The client is constructed first and the handler picks it up.
  - `client.trace(...)` - the SDK is OpenTelemetry-based now and has no
    `trace` method at all. Observations are spans, created with
    `start_observation(...)`.

Because the import was inside a function guarded by "are keys set", and
the keys are unset by default, this never raised in the test suite or in
CI. It would have raised ModuleNotFoundError on the first request of the
first deployment that actually configured tracing - the one situation
where you're relying on observability to tell you what went wrong.

Fixed at the same time: trace grouping. Nothing established a parent
span, so each LLM call became its own top-level trace and a single
question showed up in Langfuse as three unrelated traces - you could not
tell which critique belonged to which answer. question_trace() is now
the parent span, and the LangChain callbacks and tool spans nest inside
it.

What the test suite verifies (see tests/test_tracing.py) is that every
function here behaves identically - returns empty, no-ops, never raises -
whether or not tracing is configured, so the rest of the pipeline never
needs an `if tracing_enabled:` branch of its own. Live export still
needs real keys to verify.
"""
from contextlib import contextmanager

from src.config import config

_callback_handler = None
_langfuse_client = None


def _tracing_configured() -> bool:
    return bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)


def _client():
    """
    The Langfuse client singleton, or None if tracing isn't configured or
    the SDK can't be initialised. Constructing it registers the
    credentials that both the LangChain handler and the manual spans
    then use, so everything else here goes through this.
    """
    global _langfuse_client
    if not _tracing_configured():
        return None
    if _langfuse_client is None:
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
            )
        except Exception:
            # A broken or incompatible tracing SDK must not take down the
            # pipeline it's meant to observe.
            return None
    return _langfuse_client


def get_callbacks() -> list:
    """
    LangChain callback list to pass into .invoke(config={"callbacks": ...})
    for LLM calls. Always returns a list (empty when tracing isn't
    configured) so call sites never need a conditional of their own.
    """
    global _callback_handler
    if _client() is None:
        return []

    if _callback_handler is None:
        try:
            from langfuse.langchain import CallbackHandler

            _callback_handler = CallbackHandler()
        except Exception:
            return []
    return [_callback_handler]


@contextmanager
def question_trace(question: str):
    """
    One parent span per user question. Everything the graph does for that
    question - both LLM calls and the web search tool span - nests
    underneath it, so a trace reads as one unit of work.

    Yields either a Langfuse span or None, and never raises: an
    unconfigured or failing tracer yields None and the caller's body runs
    exactly as it would have.
    """
    client = _client()
    if client is None:
        yield None
        return

    try:
        with client.start_as_current_observation(
            name="lacuna_question", as_type="chain", input={"question": question}
        ) as span:
            yield span
    except Exception:
        yield None


def log_tool_span(name: str, input_data: dict, output_data: dict) -> None:
    """
    Manual span for non-LLM tool calls - the web search step isn't a
    LangChain LLM call, so it doesn't go through a callback handler.
    Any failure here (e.g. a network error reaching Langfuse) is
    swallowed deliberately: a broken trace call should never take down
    the actual pipeline it's supposed to be observing.
    """
    client = _client()
    if client is None:
        return
    try:
        client.start_observation(
            name=name, as_type="tool", input=input_data, output=output_data
        ).end()
    except Exception:
        pass


def flush() -> None:
    """
    Force-export buffered spans. Langfuse batches in a background
    thread, which is right for a long-lived server but wrong for a
    short-lived CLI or eval run that exits before the next flush
    interval - those traces would simply never arrive.
    """
    client = _client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass
