"""
Verifies tracing.py behaves safely when Langfuse isn't configured, which
is the default state in this test environment. These tests don't verify
live export works; they verify its ABSENCE never breaks anything, which
is the actual load-bearing property the rest of the pipeline depends on.

They also now verify the configured path far enough to catch the bug
this file shipped with. The old code imported `langfuse.callback`, a
module that no longer exists, from inside a function guarded by "are the
keys set" - so with keys unset it never ran, and the test suite and CI
both passed. It would have raised ModuleNotFoundError on the first
request of the first deployment that actually turned tracing on. A fake
client is injected below so the configured branch is exercised without
credentials, which is what would have caught it.
"""
import pytest

import src.tracing as tracing


@pytest.fixture(autouse=True)
def reset_tracing_singletons():
    """The client and handler are module-level caches; clear them so one
    test's configuration can't leak into the next."""
    tracing._langfuse_client = None
    tracing._callback_handler = None
    yield
    tracing._langfuse_client = None
    tracing._callback_handler = None


@pytest.fixture
def unconfigured(monkeypatch):
    monkeypatch.setattr("src.config.config.LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "")


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr("src.config.config.LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "sk-test")


class FakeSpan:
    def __init__(self):
        self.ended = False

    def end(self):
        self.ended = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeClient:
    def __init__(self):
        self.observations = []
        self.flushed = False

    def start_observation(self, **kwargs):
        self.observations.append(kwargs)
        return FakeSpan()

    def start_as_current_observation(self, **kwargs):
        self.observations.append(kwargs)
        return FakeSpan()

    def flush(self):
        self.flushed = True


class ExplodingClient:
    def start_observation(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def start_as_current_observation(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def flush(self):
        raise RuntimeError("langfuse is down")


# --- unconfigured: everything no-ops ---

def test_get_callbacks_returns_empty_list_when_unconfigured(unconfigured):
    assert tracing.get_callbacks() == []


def test_log_tool_span_is_a_no_op_when_unconfigured(unconfigured):
    tracing.log_tool_span("test_span", {"in": 1}, {"out": 2})


def test_question_trace_yields_none_when_unconfigured(unconfigured):
    with tracing.question_trace("a question") as span:
        assert span is None


def test_flush_is_a_no_op_when_unconfigured(unconfigured):
    tracing.flush()


def test_tracing_configured_requires_both_keys(monkeypatch):
    monkeypatch.setattr("src.config.config.LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "")
    assert tracing._tracing_configured() is False

    monkeypatch.setattr("src.config.config.LANGFUSE_SECRET_KEY", "sk-test")
    assert tracing._tracing_configured() is True


# --- configured: the branch that used to import a module that was gone ---

def test_question_trace_opens_one_span_per_question(configured):
    client = FakeClient()
    tracing._langfuse_client = client

    with tracing.question_trace("What is the refund policy?") as span:
        assert span is not None

    assert len(client.observations) == 1
    assert client.observations[0]["name"] == "lacuna_question"
    assert client.observations[0]["input"] == {"question": "What is the refund policy?"}


def test_log_tool_span_records_a_tool_observation(configured):
    client = FakeClient()
    tracing._langfuse_client = client

    tracing.log_tool_span("web_search", {"query": "x"}, {"results": []})

    assert client.observations[0]["as_type"] == "tool"
    assert client.observations[0]["name"] == "web_search"


def test_flush_forwards_to_the_client(configured):
    client = FakeClient()
    tracing._langfuse_client = client
    tracing.flush()
    assert client.flushed is True


def test_get_callbacks_uses_the_current_langchain_import_path(configured):
    # The regression guard: `langfuse.callback` was removed in v3. If the
    # import path is wrong again, this returns [] instead of a handler.
    tracing._langfuse_client = FakeClient()
    assert len(tracing.get_callbacks()) == 1


# --- configured but broken: tracing must never take down the pipeline ---

def test_question_trace_survives_a_failing_client(configured):
    tracing._langfuse_client = ExplodingClient()
    with tracing.question_trace("q") as span:
        assert span is None


def test_log_tool_span_survives_a_failing_client(configured):
    tracing._langfuse_client = ExplodingClient()
    tracing.log_tool_span("web_search", {}, {})


def test_flush_survives_a_failing_client(configured):
    tracing._langfuse_client = ExplodingClient()
    tracing.flush()
