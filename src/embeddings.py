"""
A local embedding model: ONNX runtime, CPU-only, no API key, no PyTorch.

This used to be langchain_community.embeddings.FastEmbedEmbeddings. Two
reasons it's a dozen lines of our own instead:

  - langchain-community is a very large transitive dependency to carry
    for one adapter class, and its embeddings module is the part of the
    package being actively thinned out. Depending on it meant depending
    on the whole thing.
  - The old wrapper hardcoded the model name, so the documented
    EMBEDDING_MODEL setting in .env.example did nothing at all. This one
    reads it.

The LangChain Embeddings interface is two methods, and fastembed's own
API is already the hard part, so there is nothing of substance being
reimplemented here.
"""
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from src.config import config


class FastEmbedEmbeddings(Embeddings):
    """Embeddings backed by a local fastembed ONNX model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        # Loaded on first use, not in __init__: constructing this object
        # is cheap and happens during app setup, but loading the model
        # downloads ~80MB on a cold cache. Deferring it keeps import time
        # and test collection fast.
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in self.model.embed(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        # query_embed applies the model's query prefix where it has one
        # (bge models are trained with an asymmetric query/passage
        # prompt, and skipping the prefix measurably costs recall).
        return next(iter(self.model.query_embed([text]))).tolist()


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str = None) -> FastEmbedEmbeddings:
    """
    Cached per model name, so the API process, the ingestion script and
    the tests all share one loaded model rather than paying the load cost
    per call site.
    """
    return FastEmbedEmbeddings(model_name=model_name or config.EMBEDDING_MODEL)
