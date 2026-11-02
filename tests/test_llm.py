"""
Tests for the shared structured-output path.

These matter more than they look. invoke_structured() is the single
retry mechanism behind every LLM call in the project, so its failure
behaviour decides what the user sees when a provider rate-limits, when a
key is wrong, and when a model returns something that doesn't fit the
schema. None of that needs a real provider to test - a fake chat model
that fails on demand covers all three.
"""
import pytest
from pydantic import BaseModel, Field, ValidationError

from src import llm as llm_module
from src.llm import MissingAPIKeyError, StubLLM, get_llm, invoke_structured
from src.schemas import CompletenessVerdict


class Shape(BaseModel):
    value: int = Field(ge=0, le=10)


class FakeStructured:
    """Fails the first `failures` calls, then succeeds."""

    def __init__(self, error, failures):
        self.error = error
        self.failures = failures
        self.prompts = []

    def invoke(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if len(self.prompts) <= self.failures:
            raise self.error
        return Shape(value=1)


class FakeLLM:
    def __init__(self, error=None, failures=0):
        self.structured = FakeStructured(error or ValueError("bad schema"), failures)

    def with_structured_output(self, schema):
        return self.structured


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Backoff is real but nobody needs to wait for it in a unit test."""
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)


# --- provider selection ---

def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "no-such-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        get_llm()


def test_groq_without_a_key_raises_an_actionable_error(monkeypatch):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "groq")
    monkeypatch.setattr("src.config.config.GROQ_API_KEY", "")
    with pytest.raises(MissingAPIKeyError, match="console.groq.com"):
        get_llm()


def test_gemini_without_a_key_raises_an_actionable_error(monkeypatch):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "gemini")
    monkeypatch.setattr("src.config.config.GOOGLE_API_KEY", "")
    with pytest.raises(MissingAPIKeyError, match="aistudio.google.com"):
        get_llm()


def test_stub_provider_needs_no_key(monkeypatch):
    monkeypatch.setattr("src.config.config.LLM_PROVIDER", "stub")
    assert isinstance(get_llm(), StubLLM)


# --- retry behaviour ---

def test_succeeds_without_retrying_when_the_first_attempt_works():
    fake = FakeLLM(failures=0)
    assert invoke_structured(fake, Shape, "prompt").value == 1
    assert len(fake.structured.prompts) == 1


def test_retries_a_validation_failure_and_succeeds():
    fake = FakeLLM(ValidationError.from_exception_data("Shape", []), failures=1)
    assert invoke_structured(fake, Shape, "prompt", max_attempts=3).value == 1
    assert len(fake.structured.prompts) == 2


def test_retry_tells_the_model_what_was_wrong():
    # Re-sending a byte-identical prompt to a near-deterministic model
    # mostly reproduces the same failure. The second attempt has to be a
    # different request to be worth making.
    fake = FakeLLM(ValueError("value must be <= 10"), failures=1)
    invoke_structured(fake, Shape, "original prompt", max_attempts=2)

    first, second = fake.structured.prompts
    assert first == "original prompt"
    assert "original prompt" in second
    assert "value must be <= 10" in second


def test_gives_up_after_max_attempts():
    fake = FakeLLM(ValueError("still broken"), failures=99)
    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        invoke_structured(fake, Shape, "prompt", max_attempts=3)
    assert len(fake.structured.prompts) == 3


def test_a_transient_error_is_retried():
    fake = FakeLLM(RuntimeError("429 Too Many Requests"), failures=1)
    assert invoke_structured(fake, Shape, "prompt", max_attempts=3).value == 1


def test_a_permanent_error_is_not_retried():
    # Retrying a bad API key three times just makes the user wait three
    # times as long for the same answer.
    fake = FakeLLM(RuntimeError("401 Unauthorized: invalid api key"), failures=99)
    with pytest.raises(MissingAPIKeyError):
        invoke_structured(fake, Shape, "prompt", max_attempts=3)
    assert len(fake.structured.prompts) == 1


def test_transient_failures_back_off(monkeypatch):
    # Jitter is pinned, because it makes the naive assertion
    # (delays[1] > delays[0]) flaky: the delays are
    # base * 2**(n-1) * (0.5 + random()), so an unlucky pair can be
    # 1.49 then 1.01 and the test fails perhaps one run in six. Pinning
    # the random factor tests the exponential growth that's actually
    # being asserted, rather than the jitter on top of it.
    delays = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: delays.append(seconds))
    monkeypatch.setattr(llm_module.random, "random", lambda: 0.5)
    fake = FakeLLM(RuntimeError("service unavailable 503"), failures=2)
    invoke_structured(fake, Shape, "prompt", max_attempts=3)

    base = llm_module.config.LLM_BACKOFF_SECONDS
    assert delays == [base * 1.0, base * 2.0]


def test_backoff_jitter_stays_within_its_band(monkeypatch):
    # The jitter is meant to spread retries so that several nodes sharing
    # a rate limit don't collide again in lockstep. Bounds, not an exact
    # value, since the whole point is that it varies.
    delays = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: delays.append(seconds))
    fake = FakeLLM(RuntimeError("429 rate limit"), failures=1)
    invoke_structured(fake, Shape, "prompt", max_attempts=2)

    base = llm_module.config.LLM_BACKOFF_SECONDS
    assert len(delays) == 1
    assert 0.5 * base <= delays[0] <= 1.5 * base


def test_validation_failures_do_not_back_off(monkeypatch):
    delays = []
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: delays.append(seconds))
    fake = FakeLLM(ValueError("schema mismatch"), failures=1)
    invoke_structured(fake, Shape, "prompt", max_attempts=2)
    assert delays == []


# --- error classification ---

@pytest.mark.parametrize("message", [
    "Rate limit reached", "429 too many requests", "connection reset",
    "503 service unavailable", "request timed out",
])
def test_transient_markers_are_classified_as_transient(message):
    assert llm_module._classify(RuntimeError(message)) == "transient"


@pytest.mark.parametrize("message", [
    "invalid api key", "401 Unauthorized", "permission denied",
    "model_not_found", "quota exceeded",
])
def test_permanent_markers_are_classified_as_permanent(message):
    assert llm_module._classify(RuntimeError(message)) == "permanent"


def test_anything_else_is_treated_as_a_schema_problem():
    assert llm_module._classify(ValueError("field required")) == "validation"


# --- the stub provider ---

def test_stub_returns_a_schema_valid_verdict():
    structured = StubLLM().with_structured_output(CompletenessVerdict)
    verdict = structured.invoke("Question: what is the price\n---\ncontext\n---\n")
    assert isinstance(verdict, CompletenessVerdict)


def test_stub_refuses_an_unknown_schema():
    with pytest.raises(NotImplementedError):
        StubLLM().with_structured_output(Shape).invoke("prompt")
