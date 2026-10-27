"""
Verifies tracing.py behaves safely when Langfuse isn't configured, which
is the default state in this test environment (and in the sandbox this
was built in - it can't reach Langfuse's servers regardless). These
tests don't verify live tracing works; they verify its ABSENCE never
breaks anything, which is the actual load-bearing property the rest of
the pipeline depends on.
"""
import src.tracing as tracing


def test_get_callbacks_returns_empty_list_when_unconfigured(monkeypatch):
    monkeypatch.setattr("src.config.config.LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "")
    assert tracing.get_callbacks() == []


def test_log_tool_span_is_a_no_op_when_unconfigured(monkeypatch):
    monkeypatch.setattr("src.config.config.LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "")
    # Should not raise, should not require network access
    tracing.log_tool_span("test_span", {"in": 1}, {"out": 2})


def test_tracing_configured_requires_both_keys(monkeypatch):
    monkeypatch.setattr("src.config.config.LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "")
    assert tracing._tracing_configured() is False

    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "sk-test")
    assert tracing._tracing_configured() is True
