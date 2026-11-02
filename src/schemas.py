"""
The structured contract the LLM must fill in for every answer.

Why this exists: free-text answers can't be programmatically checked for
completeness or routed by downstream logic. By forcing the LLM through
this schema, an "I'm not sure" becomes a machine-readable signal
(confidence + missing_info) instead of buried, easy-to-miss hedging
language inside a paragraph.

The constraints here are load-bearing, not decoration. A field the
schema merely *describes* as required is a suggestion; a field a
validator rejects is a contract, and because every LLM call goes through
llm.invoke_structured(), a rejection is automatically re-prompted with
the validation error attached. So a constraint expressed here turns into
a self-correcting behaviour for free.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class RAGResponse(BaseModel):
    answer: str = Field(
        description=(
            "The answer to the user's question, based ONLY on the provided "
            "context. If the context is insufficient, give the best partial "
            "answer possible and be explicit about what's missing."
        )
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How well the retrieved context actually supports this answer. "
            "0.9-1.0: the context directly and completely answers the "
            "question. 0.5-0.8: the context partially answers it or "
            "requires some inference. 0.0-0.4: the context barely relates "
            "to the question, or doesn't address it at all. "
            "This must reflect the CONTEXT's coverage of the question, not "
            "how confident you sound - a fluent guess is still low "
            "confidence if the context doesn't back it up."
        ),
    )

    missing_info: List[str] = Field(
        default_factory=list,
        description=(
            "Specific pieces of information that would be needed to answer "
            "more completely or confidently, but aren't present in the "
            "context. Empty list if the context fully covers the question. "
            "Be specific (e.g. 'Enterprise plan pricing tiers'), not vague "
            "(e.g. 'more information')."
        ),
    )

    used_source_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The numeric IDs (e.g. 1, 2, 3) of the [Source N] blocks in the "
            "context that were actually used to build the answer. Only "
            "include sources you actually relied on, not every source you "
            "were given. Web results are numbered in the same sequence as "
            "document chunks - cite them the same way."
        ),
    )


class CompletenessVerdict(BaseModel):
    """
    Output of an INDEPENDENT critic call - a separate LLM invocation from
    the one that generated the answer, and one that never sees the
    generator's self-reported confidence score. This split matters: an
    LLM asked to both answer and grade its own answer tends to rubber-
    stamp fluent-sounding output, so the critic only ever sees the raw
    question, context, and answer text and has to form its own judgment.
    """

    action: Literal["accept", "enrich", "ask_user"] = Field(
        description=(
            "'accept': the context directly supports the answer AND the "
            "answer actually resolves the question. 'enrich': the answer "
            "is missing specific, nameable information a web search or "
            "additional document could plausibly fill in - including the "
            "case where the answer declines to answer because the "
            "context doesn't cover the question. 'ask_user': the question "
            "itself is ambiguous or underspecified in a way no amount of "
            "extra retrieval would fix."
        )
    )

    answer_resolves_question: bool = Field(
        description=(
            "True only if the answer actually tells the user what they "
            "asked for. False if the answer instead reports that the "
            "information is absent, unavailable, or not covered by the "
            "context - however well-written and however correctly it "
            "says so. An answer that says 'the context does not state X' "
            "is a refusal, not an answer, and must be False."
        )
    )

    sufficiency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How well the context supports the answer, independently assessed.",
    )

    gap_description: Optional[str] = Field(
        default=None,
        description=(
            "REQUIRED if action is 'enrich' or 'ask_user'. Must name the "
            "specific missing fact or ambiguity - not a vague restatement "
            "of the question - so it's actually useful as a search query "
            "or a clarifying question later."
        ),
    )

    reasoning: str = Field(
        description="One or two sentences justifying the verdict."
    )

    @model_validator(mode="after")
    def _accept_requires_a_resolved_question(self):
        """
        The defect this closes, which only showed up once the end-to-end
        benchmark was actually run: every single unanswerable question
        came back as `accepted`, at sufficiency 1.0.

        Nothing was fabricating. The Answerer correctly said "the context
        does not state who the CTO is" at confidence 0.0. The critic was
        asked whether the context supported that answer - and it does,
        completely, so it accepted. Both components were right about the
        question they were each asked.

        The conflation was in `accept` itself, which was carrying two
        claims at once: "this answer is well-supported" and "this
        question has been answered". A refusal is maximally
        well-supported and entirely unanswered, so it satisfied the only
        one of those that was ever checked, and the caller received a
        refusal labelled as a finished answer. The enrichment loop never
        ran either, because the gap was never routed anywhere - two
        enrichment rounds across fourteen questions.

        Separating the two claims is what makes the loop work: a refusal
        now routes to enrichment, which is exactly the case web search
        exists for.
        """
        if self.action == "accept" and not self.answer_resolves_question:
            raise ValueError(
                "action 'accept' requires answer_resolves_question=True. "
                "The answer reports that the information is absent rather "
                "than supplying it, so the question is unresolved: choose "
                "'enrich' if a search could plausibly find it, or "
                "'ask_user' if the question itself is ambiguous."
            )
        return self

    @model_validator(mode="after")
    def _gap_required_for_non_accept(self):
        """
        'enrich' without a gap_description is the single most damaging
        thing the critic can return: the enrichment node has nothing to
        search for, so it falls back to re-searching the raw question -
        which is exactly the search that already failed to help. Rejecting
        it here means invoke_structured() re-prompts with the error, and
        the loop gets a real search query instead of a wasted round.
        """
        if self.action in ("enrich", "ask_user") and not (self.gap_description or "").strip():
            raise ValueError(
                f"gap_description is required when action is '{self.action}': "
                f"name the specific missing fact or the specific ambiguity."
            )
        return self


class Citation(BaseModel):
    """
    One resolved source behind an answer.

    Previously an answer's sources were a plain list of filenames, built
    by mapping [Source N] ids to document metadata. That silently dropped
    every web citation: the web results were numbered into the same
    [Source N] sequence the model cites from, but the resolver only
    indexed into the retrieved *documents*, so an id pointing at a web
    result fell outside the range and was discarded. The answer cited a
    source the caller never saw.

    Making both kinds one type fixes that and makes provenance explicit:
    a caller can tell at a glance whether a claim came from the uploaded
    corpus or from the open web, which matters a great deal more for a
    web result.
    """

    kind: Literal["document", "web"]
    label: str  # filename for a document, page title for a web result
    url: Optional[str] = None  # web results only
    source_id: int  # the [Source N] number the model actually cited
