"""
The Completeness-checker: an INDEPENDENT critic call, separate from the
one that generated the answer.

Why separate: an LLM asked to both answer a question and grade its own
answer tends toward sycophancy - it rates fluent-sounding output as
trustworthy even when the underlying context barely supports it. This
call deliberately never sees the generator's self-reported confidence
score, only the raw question, context, and answer text, so it has to form
an independent judgment rather than rubber-stamping one.

What it DOES now see is the enrichment history. Without it the critic was
stateless across loop iterations, and the failure that caused was
concrete: it would return the same gap_description in round 2 that it
returned in round 1, the enrichment node would run a search it had
already run, and the iteration budget would be spent on a duplicate
request. A critic that knows what has already been tried can either name
a *different* gap or stop asking.

The prompt is careful about one thing in particular. Telling a model
"searching has not been productive" is an invitation to give up and
accept, and accepting an unsupported answer is the exact failure this
whole project exists to prevent. So the history is framed as "don't
repeat a query", never as "stop holding out for evidence", and the
instruction not to accept on fatigue is stated explicitly. Running out
of budget is the orchestrator's decision to make, not the critic's.
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
{clarifications}{history}
You are judging TWO separate things, and must not let one stand in for \
the other:

1. Does the retrieved context support what the answer claims?
2. Does the answer actually resolve the question that was asked?

These come apart in one specific and important case. If the answer says \
"the context does not state X", that is perfectly supported by the \
context - and it resolves nothing. It is a refusal, not an answer. Set \
answer_resolves_question=false for it and do NOT choose "accept"; the \
missing fact is precisely what a web search exists to go and find, so \
choose "enrich" and name the missing fact as the gap. Reserve "accept" \
for answers that genuinely tell the user what they asked for.

Do not judge writing quality, fluency, or tone - a well-written answer built \
on thin context is still a bad answer for this review, and a fluent \
refusal is still a refusal.

Context blocks marked `web:` came from a general web search. Treat them as \
weak evidence about the question's specific subject: a web result \
describing something in general terms does not establish a fact about this \
particular product, company, or document set. If the answer leans on a web \
result for a claim about the question's specific subject, that is a gap, \
not support.

Guidelines:
- "accept": the context directly supports the answer with no material \
gaps, AND the answer resolves the question.
- "enrich": the answer is missing specific, nameable information that a \
web search or additional document could plausibly fill in - including \
every case where the answer declines because the context doesn't cover \
the question.
- "ask_user": the question itself is ambiguous or underspecified in a way \
no amount of extra retrieval would fix (e.g. missing a date range, unclear \
which product/entity is meant).

gap_description is REQUIRED for "enrich" and "ask_user" - name the exact \
missing fact, figure, or ambiguity, not a vague restatement of the \
question. Further searches have a real cost, so only choose "enrich" when \
there's a concrete, findable gap - not simply because the answer could \
theoretically be more detailed.

Never choose "accept" merely because earlier searching was unproductive. If \
the context still does not support the answer, say so; something else \
decides when to stop looking."""

HISTORY_TEMPLATE = """
Searches already run for this question, which did NOT close the gap:
{searches}
Do not request any of these again. If you choose "enrich", gap_description \
must describe something materially different from the above.
"""


def format_history(enrichment_log: list) -> str:
    """
    Render the enrichment log for the prompt. Pure function (no LLM call)
    so the formatting is unit-testable on its own, same reason
    rag.resolve_citations is pulled out.
    """
    if not enrichment_log:
        return ""
    lines = []
    for entry in enrichment_log:
        query = entry.get("query", "")
        if entry.get("error"):
            outcome = f"search failed ({entry['error']})"
        else:
            outcome = f"{entry.get('num_results', 0)} result(s)"
        lines.append(f"  {entry.get('iteration', '?')}. \"{query}\" -> {outcome}")
    return HISTORY_TEMPLATE.format(searches="\n".join(lines))


def check_completeness(
    question: str,
    context: str,
    answer_text: str,
    enrichment_log: list = None,
    clarifications: list = None,
    max_attempts: int = None,
) -> CompletenessVerdict:
    """
    Runs the independent critic call. Raises MissingAPIKeyError or
    RuntimeError on failure, same contract as rag.generate_answer(), so
    both can be handled identically by whatever calls them.

    The critic sees any clarifications the user has already given, for
    the same reason it sees the enrichment history: without them it is
    stateless across turns and will ask the user to disambiguate
    something they have already disambiguated, which is the most
    irritating failure a clarification loop can have.
    """
    from src.rag import format_clarifications

    prompt = COMPLETENESS_CHECK_PROMPT.format(
        question=question,
        context=context,
        answer=answer_text,
        clarifications=format_clarifications(clarifications or []),
        history=format_history(enrichment_log or []),
    )
    llm = get_llm()
    return invoke_structured(
        llm,
        CompletenessVerdict,
        prompt,
        max_attempts=max_attempts,
        callbacks=tracing.get_callbacks(),
    )
