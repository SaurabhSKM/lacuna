"""
Optional Langfuse tracing. Every function here degrades to a safe no-op
if LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't set - tracing is
strictly additive, never a requirement for the pipeline to run, same
philosophy as the LLM provider and web search tool: absence of
credentials degrades gracefully rather than crashing anything.

This sandbox can't reach Langfuse's servers to verify live tracing
end-to-end (same network limitation as Groq, Hugging Face, and
DuckDuckGo before it). What IS verified (see tests/test_tracing.py) is
that every function here behaves identically - returns empty/no-ops -
whether or not tracing is configured, so the rest of the pipeline never
needs an `if tracing_enabled:` branch of its own.

Known limitation: each node's LLM call currently becomes its own
top-level trace in Langfuse rather than all calls for one user question
being grouped under a single trace. Worth revisiting once there's live
access to see whether that grouping actually matters for debugging in
practice, rather than building it blind.
"""
from src.config import config

_callback_handler = None
_langfuse_client = None


def _tracing_configured() -> bool:
    return bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)


def get_callbacks() -> list:
    """
    LangChain callback list to pass into .invoke(config={"callbacks": ...})
    for LLM calls. Always returns a list (empty when tracing isn't
    configured) so call sites never need a conditional of their own.
    """
    global _callback_handler
    if not _tracing_configured():
        return []

    if _callback_handler is None:
        from langfuse.callback import CallbackHandler

        _callback_handler = CallbackHandler(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
    return [_callback_handler]


def log_tool_span(name: str, input_data: dict, output_data: dict) -> None:
    """
    Manual trace for non-LLM tool calls - the web search step isn't a
    LangChain LLM call, so it doesn't go through a callback handler.
    Any failure here (e.g. a network error reaching Langfuse) is
    swallowed deliberately: a broken trace call should never take down
    the actual pipeline it's supposed to be observing.
    """
    if not _tracing_configured():
        return

    global _langfuse_client
    try:
        if _langfuse_client is None:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY,
                secret_key=config.LANGFUSE_SECRET_KEY,
                host=config.LANGFUSE_HOST,
            )
        _langfuse_client.trace(name=name, input=input_data, output=output_data)
    except Exception:
        pass
