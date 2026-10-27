"""
The structured contract the LLM must fill in for every answer.

Why this exists: free-text answers can't be programmatically checked for
completeness or routed by downstream logic. By forcing the LLM through
this schema, an "I'm not sure" becomes a machine-readable signal
(confidence + missing_info) instead of buried, easy-to-miss hedging
language inside a paragraph.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


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
            "were given."
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
            "'accept': the context directly supports the answer with no "
            "material gaps. 'enrich': the answer is missing specific, "
            "nameable information a web search or additional document "
            "could plausibly fill in. 'ask_user': the question itself is "
            "ambiguous or underspecified in a way no amount of extra "
            "retrieval would fix."
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
            "Required if action is 'enrich' or 'ask_user'. Must name the "
            "specific missing fact or ambiguity - not a vague restatement "
            "of the question - so it's actually useful as a search query "
            "or a clarifying question later."
        ),
    )

    reasoning: str = Field(
        description="One or two sentences justifying the verdict."
    )
