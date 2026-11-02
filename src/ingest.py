"""
Ingestion pipeline: raw files -> chunks -> embeddings -> Chroma vector store.

This is the "unglamorous but essential" part of RAG. The stages are kept
as separate functions on purpose so each is easy to test and reason about
independently:

  1. load_documents    - read files off disk into LangChain Document objects
  2. chunk_documents   - split long documents into retrieval-sized pieces
  3. build_vectorstore - embed the chunks and persist them to Chroma

The loaders are local functions rather than langchain_community's
DirectoryLoader / TextLoader / PyPDFLoader. Those three classes were the
only thing the project used langchain-community for, and "glob a
directory, read text, read a PDF with pypdf" is a smaller amount of code
than the dependency was worth. It also means a PDF page number lands in
the metadata in the shape this project actually wants, rather than the
shape the loader happened to pick.
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import config
from src.embeddings import get_embedding_model

TEXT_EXTENSIONS = (".txt", ".md")
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS + (".pdf",)


def _load_text_file(path: Path) -> list:
    # errors="replace" rather than strict: one byte of bad encoding in a
    # 200-page export shouldn't abort the whole ingestion run.
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": str(path)})]


def _load_pdf_file(path: Path) -> list:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue  # scanned/image-only page: nothing to embed
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "page": page_number},
            )
        )
    return docs


def load_file(file_path) -> list:
    """Load one file by extension. Returns a list of Documents (a PDF
    yields one per page with text on it)."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        return _load_text_file(path)
    if ext == ".pdf":
        return _load_pdf_file(path)
    raise ValueError(
        f"Unsupported file extension '{ext}'. "
        f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


# Kept under its original name: the /upload endpoint and the tests call it.
load_single_document = load_file


def load_documents(docs_dir: str) -> list:
    """
    Load every .txt, .md, and .pdf file under docs_dir into a list of
    LangChain Document objects (each has .page_content and .metadata).
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents = []
    skipped = []
    # sorted() so ingestion order - and therefore chunk ids and BM25
    # ordering - is reproducible rather than filesystem-dependent.
    for path in sorted(docs_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            documents.extend(load_file(path))
        except Exception as e:
            # One unreadable file shouldn't sink the whole run; report it
            # and carry on, so a 500-document ingestion isn't lost to a
            # single corrupt PDF.
            skipped.append(f"{path.name}: {e}")

    if skipped:
        print(f"Skipped {len(skipped)} unreadable file(s):")
        for s in skipped:
            print(f"  - {s}")

    if not documents:
        raise ValueError(
            f"No readable .txt, .md, or .pdf files found in {docs_dir}. "
            "Nothing to ingest."
        )

    return documents


def chunk_documents(documents):
    """
    Split documents into overlapping chunks. Overlap matters: if a fact
    sits right at a chunk boundary, overlap keeps it from being severed
    away from its surrounding context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, persist_dir: str):
    """
    Embed all chunks and persist them to a Chroma collection on disk.
    Running this again with the same persist_dir will add to the
    existing collection.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(persist_dir: str):
    """Reopen an existing Chroma collection for querying."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedding_model(),
    )


def ingest(docs_dir: str, persist_dir: str):
    """End-to-end: load -> chunk -> embed -> persist. Returns the store."""
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunk(s).")
    vectorstore = build_vectorstore(chunks, persist_dir)
    print(f"Vector store persisted to: {persist_dir}")
    return vectorstore
