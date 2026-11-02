"""
The enrichment tool: a free, no-API-key web search (DuckDuckGo), plus
the rules for writing useful findings back into the knowledge base.

The import is `ddgs`, not `duckduckgo_search`. The package was renamed;
the old distribution is now a deprecation shim that re-exports this one
and warns on import, and the pinned 6.x version this project had
predates the rename entirely. The result dict shape (`title`, `href`,
`body`) is unchanged, so nothing downstream had to move.

The pure functions here - dedupe_results, format_web_results,
select_persistable - have no network call and no vector store inside
them, which is why they carry the actual logic. The live search is a
thin wrapper around someone else's client; the decisions are all
testable without the network.
"""
from langchain_core.documents import Document

from src.config import config


def web_search(query: str, max_results: int = 3) -> list:
    """
    Returns a list of dicts: {"title": ..., "href": ..., "body": ...}.
    Raises whatever the client raises on failure (e.g. rate limiting) -
    the enrichment node is responsible for catching that and degrading
    gracefully rather than crashing the whole graph over a flaky search.
    """
    from ddgs import DDGS

    return list(DDGS().text(query, max_results=max_results))


def dedupe_results(existing: list, new: list) -> list:
    """
    Return the entries of `new` not already present in `existing`,
    deduped against each other too, keyed on URL.

    Enrichment rounds overlap heavily in practice: two different gap
    descriptions about the same subject return many of the same pages.
    Without this, the same result was appended to the context again under
    a *new* [Source N] number, so the model saw one page as two
    independent sources agreeing with each other - which is precisely the
    kind of false corroboration that makes a thin answer look supported.
    It also wasted context window and, on accept, wrote duplicate chunks
    into the store.
    """
    seen = {r.get("href") for r in existing if r.get("href")}
    out = []
    for result in new:
        url = result.get("href")
        if url:
            if url in seen:
                continue
            seen.add(url)
        elif not (result.get("body") or "").strip():
            continue  # no url and no text: nothing of value
        out.append(result)
    return out


def format_web_results(results: list, start_index: int) -> str:
    """
    Format search results as numbered [Source N] blocks, continuing the
    numbering after whatever document sources already exist.

    Retained for callers that want just the web block on its own; the
    graph builds its context through rag.build_source_refs(), which
    numbers documents and web results into one sequence and keeps the
    mapping needed to resolve a citation back to a URL.
    """
    parts = []
    for i, r in enumerate(results, start=start_index):
        title = r.get("title", "untitled")
        body = r.get("body", "")
        url = r.get("href", "")
        parts.append(f"[Source {i}] (web: {title} - {url})\n{body}")
    return "\n\n".join(parts)


def select_persistable(web_sources: list, cited_urls: set, only_cited: bool = None) -> list:
    """
    Decide which web results are worth keeping permanently.

    The rule is "the answer actually used it", not "the search returned
    it". A search for one specific gap returns three results; typically
    one contains the missing fact and the other two are noise that
    happened to rank. Persisting all three put two chunks of unvetted
    web text into the knowledge base for every gap closed, where they
    then compete with the user's own documents in every future
    retrieval. Keeping only cited results means the store grows by
    exactly what earned its place.

    Set PERSIST_ONLY_CITED=false to keep the old behaviour of persisting
    everything that was searched.
    """
    only_cited = config.PERSIST_ONLY_CITED if only_cited is None else only_cited
    out = []
    for r in web_sources:
        if not (r.get("body") or "").strip():
            continue
        if only_cited and r.get("href") not in cited_urls:
            continue
        out.append(r)
    return out


def _already_stored_urls(vectorstore, urls: list) -> set:
    """
    Which of these URLs are already in the collection.

    Checked by metadata rather than by embedding similarity: an exact URL
    match is both cheaper and a stronger signal than a similarity score
    near a threshold. Metadata filtering isn't available on every store,
    so a failure here degrades to "assume nothing is stored" - at worst
    that re-adds a chunk, which is the behaviour we had before.
    """
    if not urls:
        return set()
    try:
        found = vectorstore.get(
            where={"source": {"$in": [f"web:{u}" for u in urls]}},
            include=["metadatas"],
        )
    except Exception:
        return set()

    stored = set()
    for meta in found.get("metadatas") or []:
        source = (meta or {}).get("source", "")
        if source.startswith("web:"):
            stored.add(source[len("web:"):])
    return stored


def persist_web_sources(
    vectorstore, web_sources: list, origin_question: str, cited_urls: set = None
) -> int:
    """
    Permanently add web search results that helped produce an ACCEPTED
    answer into the knowledge base, so the next question on the same
    topic doesn't need to re-search the web at all. This is what makes
    the enrichment loop actually close a gap rather than just patch over
    it for one answer.

    Deliberately only called from the accept path (see graph.py), never
    from a capped or ask_user outcome - search results that didn't
    actually lead to a confident answer aren't worth polluting the
    knowledge base with. Two filters narrow it further: only results the
    answer cited (select_persistable), and only URLs not already stored.

    `cited_urls` of None means "no citation information available" and
    falls back to persisting every non-empty result, so a caller that
    can't resolve citations still gets the old behaviour instead of
    silently persisting nothing.
    """
    only_cited = None if cited_urls is not None else False
    candidates = select_persistable(web_sources, cited_urls or set(), only_cited=only_cited)
    if not candidates:
        return 0

    already = _already_stored_urls(
        vectorstore, [r.get("href") for r in candidates if r.get("href")]
    )

    docs = []
    for r in candidates:
        url = r.get("href", "")
        if url and url in already:
            continue
        docs.append(
            Document(
                page_content=f"{r.get('title', 'untitled')}\n{r.get('body', '')}",
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
