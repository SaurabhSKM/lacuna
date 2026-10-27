"""
The Completeness-checker: an INDEPENDENT critic call, separate from the
one that generated the answer.

Why separate: an LLM asked to both answer a question and grade its own
answer tends toward sycophancy - it rates fluent-sounding output as
trustworthy even when the underlying context barely supports it. This
call deliberately never sees the generator's self-reported confidence
score, only the raw question, context, and answer text, so it has to form
an independent judgment rather than rubber-stamping one.
"""
from src.llm import get_llm, invoke_structured
from src.schemas import CompletenessVerdict
from src import tracing

COMPLETENESS_CHECK_PROMPT = """You are a strict, independent reviewer. You did \
not write the answer below - someone else did - and your job is to judge it \
critically, not to defend it.

Question: {question}

Retrieved context (what the answer-writer had available):
---
{context}
---

Answer given:
---
{answer}
---

Judge ONLY whether the retrieved context actually supports this answer. \
Do not judge writing quality, fluency, or tone - a well-written answer built \
on thin context is still a bad answer for this review.

Guidelines:
- "accept": the context directly supports the answer with no material gaps.
- "enrich": the answer is missing specific, nameable information that a \
web search or additional document could plausibly fill in.
- "ask_user": the question itself is ambiguous or underspecified in a way \
no amount of extra retrieval would fix (e.g. missing a date range, unclear \
which product/entity is meant).

Be specific in gap_description if action is "enrich" - name the exact \
missing fact or figure, not a vague restatement of the question. Further \
searches have a real cost, so only choose "enrich" when there's a concrete, \
findable gap - not simply because the answer could theoretically be more \
detailed."""


def check_completeness(
    question: str, context: str, answer_text: str, max_retries: int = 2
) -> CompletenessVerdict:
    """
    Runs the independent critic call. Raises MissingAPIKeyError or
    RuntimeError on failure, same contract as rag.generate_answer(), so
    both can be handled identically by whatever calls them.
    """
    prompt = COMPLETENESS_CHECK_PROMPT.format(
        question=question, context=context, answer=answer_text
    )
    llm = get_llm()
    return invoke_structured(
        llm, CompletenessVerdict, prompt, max_retries=max_retries, callbacks=tracing.get_callbacks()
    )
