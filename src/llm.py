"""
Thin wrapper around whichever LLM provider is configured. The point of
this file is that every other module just calls get_llm() and never
cares whether it's Groq, Gemini, or something else later - that's the
"model-agnostic" property LangChain's chat model interface gives us.

invoke_structured() is the one place structured-output validation and
retrying lives. Two things it deliberately does that a bare retry loop
doesn't:

  1. It tells the model what was wrong. Re-sending a byte-identical
     prompt to a near-deterministic model is close to a no-op; the
     second attempt fails the same way the first did. The validation
     error is appended to the prompt so attempt 2 is actually a
     different request.
  2. It distinguishes transient from permanent failures. A 429 or a
     socket timeout is worth backing off and retrying. A 401 is not -
     retrying a bad API key three times just makes the user wait three
     times as long for the same error.
"""
import random
import re
import time

from src.config import config


class MissingAPIKeyError(RuntimeError):
    pass


# Substrings that mark a failure as worth retrying. Matched against the
# lowercased exception text because the provider SDKs all raise their own
# exception types and we don't want to import every one of them here
# just to isinstance-check it.
_TRANSIENT_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "too many requests",
    "timeout",
    "timed out",
    "connection",
    "temporarily unavailable",
    "service unavailable",
    "502",
    "503",
    "504",
    "overloaded",
    "internal server error",
)

# Substrings that mark a failure as permanent: no amount of retrying
# fixes credentials or a model name that doesn't exist.
_PERMANENT_MARKERS = (
    "api key",
    "api_key",
    "unauthorized",
    "401",
    "403",
    "permission denied",
    "invalid_api_key",
    "authentication",
    "model_not_found",
    "does not exist",
    "quota",
)


def _classify(error: Exception) -> str:
    """Return "permanent", "transient", or "validation"."""
    text = str(error).lower()
    if any(m in text for m in _PERMANENT_MARKERS):
        return "permanent"
    if any(m in text for m in _TRANSIENT_MARKERS):
        return "transient"
    # Anything else is treated as a schema/parse problem, which is the
    # case worth re-prompting with feedback.
    return "validation"


def get_llm():
    provider = config.LLM_PROVIDER.lower()

    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise MissingAPIKeyError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com and add it to your .env file."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            temperature=config.LLM_TEMPERATURE,
        )

    if provider == "gemini":
        if not config.GOOGLE_API_KEY:
            raise MissingAPIKeyError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to your "
                ".env file."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            google_api_key=config.GOOGLE_API_KEY,
            model=config.GEMINI_MODEL,
            temperature=config.LLM_TEMPERATURE,
        )

    if provider == "stub":
        return StubLLM()

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'groq', 'gemini', or 'stub'."
    )


def invoke_structured(llm, schema, prompt: str, max_attempts: int = None, callbacks: list = None):
    """
    Bind `schema` (a Pydantic model) as structured output on `llm`,
    invoke `prompt`, and retry on failure.

    This is shared by every node that needs a validated object back from
    the LLM (the Answerer and the Completeness-checker) - one retry
    mechanism instead of each one reimplementing its own, which is
    exactly the kind of duplication that tends to drift out of sync.

    `callbacks`, if provided, is passed through to .invoke() - this is
    how Langfuse tracing (src/tracing.py) hooks into every structured
    call without this function needing to know tracing exists.

    Raises MissingAPIKeyError for a permanent auth failure (so callers
    can report something actionable instead of "failed after 3
    attempts"), or RuntimeError if every attempt fails.
    """
    max_attempts = max_attempts or config.LLM_MAX_ATTEMPTS
    structured_llm = llm.with_structured_output(schema)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = prompt
        if last_error is not None:
            # Feedback, so this attempt differs from the last one.
            attempt_prompt = (
                f"{prompt}\n\n---\nYour previous response could not be "
                f"parsed into the required format. The error was:\n"
                f"{_truncate(str(last_error))}\n"
                f"Return a response that satisfies every field "
                f"constraint exactly."
            )

        invoke_kwargs = {"config": {"callbacks": callbacks}} if callbacks else {}
        try:
            return structured_llm.invoke(attempt_prompt, **invoke_kwargs)
        except Exception as e:  # validation errors, malformed tool calls, etc.
            kind = _classify(e)
            last_error = e

            if kind == "permanent":
                raise MissingAPIKeyError(
                    f"The LLM provider rejected the request and retrying "
                    f"will not help: {e}"
                ) from e

            if attempt < max_attempts:
                if kind == "transient":
                    # Exponential backoff with jitter. Jitter matters when
                    # several nodes hit a shared rate limit at once -
                    # without it they retry in lockstep and collide again.
                    delay = config.LLM_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    time.sleep(delay * (0.5 + random.random()))
                else:
                    # A validation failure is immediate - no point waiting
                    # before re-prompting with the error attached.
                    pass

    raise RuntimeError(
        f"Structured output ({schema.__name__}) failed after "
        f"{max_attempts} attempt(s): {last_error}"
    ) from last_error


def _truncate(text: str, limit: int = 500) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


# --- Offline stub provider -------------------------------------------------

class _StubStructured:
    """
    Produces a schema-valid object by reading the prompt, with no model
    behind it. Its only job is to let the full graph - including the
    enrichment cycle and the iteration cap - run in tests and in CI,
    where there is no API key and no network.

    It is NOT an evaluation harness and its "answers" mean nothing. The
    behaviour it encodes is intentionally crude and keyword-driven: say
    "accept" when the context plainly contains the question's terms, say
    "ask_user" for a question with no content words of its own, otherwise
    say "enrich". That is enough to drive every branch of the graph
    deterministically, which is what the tests need.
    """

    # Questions this short and this generic carry no retrievable subject.
    _AMBIGUOUS = re.compile(
        r"^\s*(what is the price|is it certified|what happens if i cancel)\s*\??\s*$",
        re.IGNORECASE,
    )

    def __init__(self, schema):
        self.schema = schema

    def invoke(self, prompt: str, **kwargs):
        from src.schemas import RAGResponse, CompletenessVerdict

        question = _extract(prompt, "Question:")
        context = _extract_block(prompt)
        overlap = _overlap(question, context)

        if self.schema is RAGResponse:
            if not context.strip() or context.strip().startswith("(no "):
                return RAGResponse(
                    answer="[stub] No context was retrieved, so there is nothing to answer from.",
                    confidence=0.0,
                    missing_info=["any document covering this question"],
                    used_source_ids=[],
                )
            return RAGResponse(
                answer=f"[stub] Answer assembled from {len(_source_ids(context))} source(s).",
                confidence=round(min(0.95, overlap), 2),
                missing_info=[] if overlap > 0.6 else ["specific figures"],
                used_source_ids=_source_ids(context)[:2],
            )

        if self.schema is CompletenessVerdict:
            if self._AMBIGUOUS.match(question):
                return CompletenessVerdict(
                    action="ask_user",
                    sufficiency_score=0.2,
                    gap_description="[stub] the question does not say which plan or product is meant",
                    reasoning="[stub] question is underspecified",
                )
            if overlap > 0.6:
                return CompletenessVerdict(
                    action="accept",
                    sufficiency_score=round(min(0.95, overlap), 2),
                    reasoning="[stub] context covers the question terms",
                )
            return CompletenessVerdict(
                action="enrich",
                sufficiency_score=round(overlap, 2),
                gap_description=f"[stub] specific details about: {question[:80]}",
                reasoning="[stub] context only partially covers the question",
            )

        raise NotImplementedError(f"StubLLM has no behaviour for {self.schema.__name__}")


class StubLLM:
    def with_structured_output(self, schema):
        return _StubStructured(schema)


_WORD = re.compile(r"[a-z0-9]{4,}")
_STOP = {"what", "does", "which", "there", "their", "this", "that", "with", "from", "have", "aria"}


def _words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _overlap(question: str, context: str) -> float:
    qw = _words(question)
    if not qw:
        return 0.0
    return len(qw & _words(context)) / len(qw)


def _extract(prompt: str, marker: str) -> str:
    for line in prompt.splitlines():
        if line.startswith(marker):
            return line[len(marker):].strip()
    return ""


def _extract_block(prompt: str) -> str:
    """Pull out the text between the first pair of `---` fence lines."""
    parts = prompt.split("\n---\n")
    return parts[1] if len(parts) > 2 else ""


def _source_ids(context: str) -> list:
    return [int(n) for n in re.findall(r"\[Source (\d+)\]", context)]
