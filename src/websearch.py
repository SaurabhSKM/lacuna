"""
The enrichment tool: a free, no-API-key web search (DuckDuckGo).

format_web_results() is kept as a pure function - no network call inside
it - for the same reason resolve_sources() in rag.py was pulled out on
its own: it can be unit-tested with canned results, independent of
whether the live network call succeeds. That separation matters here
specifically because this sandbox's network allowlist can't reach
duckduckgo.com, so the pure formatting logic is the part actually
verified by the test suite; the live call is verified on your machine.
"""
from duckduckgo_search import DDGS
from langchain_core.documents import Document


def web_search(query: str, max_results: int = 3) -> list:
    """
    Returns a list of dicts: {"title": ..., "href": ..., "body": ...}.
    Raises whatever DDGS raises on failure (e.g. rate limiting) - the
    enrichment node is responsible for catching that and degrading
    gracefully rather than crashing the whole graph over a flaky search.
    """
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def format_web_results(results: list, start_index: int) -> str:
    """
    Format search results as numbered [Source N] blocks, continuing the
    numbering after whatever document sources already exist, so the
    LLM's used_source_ids stay consistent whether it's citing a document
    chunk or a web result.
    """
    parts = []
    for i, r in enumerate(results, start=start_index):
        title = r.get("title", "untitled")
        body = r.get("body", "")
        url = r.get("href", "")
        parts.append(f"[Source {i}] (web result)\nTitle: {title}\nURL: {url}\n{body}")
    return "\n\n".join(parts)


def persist_web_sources(vectorstore, web_sources: list, origin_question: str) -> int:
    """
    Permanently add web search results that helped produce an ACCEPTED
    answer into the knowledge base, so the next question on the same
    topic doesn't need to re-search the web at all. This is what makes
    the enrichment loop actually close a gap rather than just patch over
    it for one answer - the "self-improving" part of the project's name.

    Deliberately only called from the accept path (see graph.py), never
    from a capped or ask_user outcome - search results that didn't
    actually lead to a confident answer aren't worth polluting the
    knowledge base with.
    """
    docs = []
    for r in web_sources:
        title = r.get("title", "untitled")
        body = r.get("body", "")
        url = r.get("href", "")
        if not body:
            continue
        docs.append(
            Document(
                page_content=f"{title}\n{body}",
                metadata={
                    "source": f"web:{url}" if url else "web:unknown",
                    "auto_ingested": True,
                    "origin_question": origin_question,
                },
            )
        )

    if not docs:
        return 0

    vectorstore.add_documents(docs)
    return len(docs)
