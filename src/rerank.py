"""
Cross-encoder reranking, the precision stage of retrieval.

Dense and BM25 retrieval are both *bi-encoders* in effect: the question
and the chunk are scored independently and compared, which is what makes
them fast enough to run over a whole collection, and also what limits
them. Neither ever looks at the question and the chunk together, so
neither can tell that "which release added FIDO2" and a chunk listing
FIDO2 under release 4.1 are an exact answer, rather than merely
topically close.

A cross-encoder does look at them together - one forward pass per
(question, chunk) pair - which is far too expensive for a whole
collection and entirely affordable over the ~12 candidates that fusion
produces. So the shape is: hybrid retrieval for recall, reranking for
precision. That ordering matters; a reranker can only reorder what
retrieval already found, so it is not a substitute for BM25 getting the
rare-term chunk into the candidate pool in the first place.

The second thing it buys is a much better *gate* signal. The relevance
floor previously read vector distance, which is a weak and
metric-dependent proxy. A cross-encoder emits a calibrated-ish
relevance logit for the pair, which is the question the gate is actually
asking.

This uses fastembed's reranker, so it's the same local ONNX runtime as
the embeddings and needs no new dependency and no API key.
"""
import math
from functools import lru_cache

from src.config import config


class CrossEncoderReranker:
    """Scores (question, chunk) pairs with a local cross-encoder."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.RERANK_MODEL
        self._model = None

    @property
    def model(self):
        # Loaded on first use, like the embedding model: constructing
        # this is cheap and happens during app setup, but loading
        # downloads ~90MB on a cold cache.
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def score(self, question: str, texts: list) -> list:
        """
        Relevance in [0,1], one per text, in the order given.

        The model emits an unbounded logit, so it's squashed with a
        logistic function to get something a threshold can be set
        against. That makes the number comparable across queries, which
        the raw logit is not, and puts it on the same [0,1] scale the
        rest of the gate already speaks.
        """
        if not texts:
            return []
        logits = list(self.model.rerank(question, list(texts)))
        return [_sigmoid(x) for x in logits]


def _sigmoid(x: float) -> float:
    # Guarded against overflow for large-magnitude logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    return math.exp(max(x, -60.0)) / (1.0 + math.exp(max(x, -60.0)))


@lru_cache(maxsize=2)
def get_reranker(model_name: str = None) -> CrossEncoderReranker:
    """Cached per model name, so one loaded model is shared across the
    API process, the CLI and the eval harness."""
    return CrossEncoderReranker(model_name=model_name or config.RERANK_MODEL)
