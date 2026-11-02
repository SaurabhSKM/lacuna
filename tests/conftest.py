"""
Shared test configuration.

The whole suite is offline by design - fake embeddings, a deterministic
stub LLM provider, pure scoring functions - so that CI needs no secrets
and no network. Cross-encoder reranking is the one default that breaks
that promise: it downloads ~90MB of ONNX weights on first use. It's
disabled here for every test rather than in each test that happens to
build a retriever, so a new test can't silently reintroduce a network
dependency into CI.

Reranking is exercised for real by eval/retrieval_eval.py, which is
where the numbers in the README come from.
"""
import pytest


@pytest.fixture(autouse=True)
def offline_retrieval(monkeypatch):
    monkeypatch.setattr("src.config.config.RERANK", False)
