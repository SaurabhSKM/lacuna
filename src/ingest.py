"""
Ingestion pipeline: raw files -> chunks -> embeddings -> Chroma vector store.

This is the "unglamorous but essential" part of RAG. Three separate ideas
live in this file, kept as separate functions on purpose so each is easy
to test and reason about independently:

  1. load_documents   - read files off disk into LangChain Document objects
  2. chunk_documents   - split long documents into retrieval-sized pieces
  3. build_vectorstore  - embed the chunks and persist them to Chroma
"""
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

from src.config import config


def load_documents(docs_dir: str):
    """
    Load every .txt, .md, and .pdf file under docs_dir into a list of
    LangChain Document objects (each has .page_content and .metadata).
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents = []

    # Plain text / markdown
    text_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents.extend(text_loader.load())

    md_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents.extend(md_loader.load())

    # PDFs need their own loader
    for pdf_path in docs_path.glob("**/*.pdf"):
        documents.extend(PyPDFLoader(str(pdf_path)).load())

    if not documents:
        raise ValueError(
            f"No .txt, .md, or .pdf files found in {docs_dir}. "
            "Nothing to ingest."
        )

    return documents


def load_single_document(file_path: str):
    """
    Load exactly one file (used by the /upload API endpoint, where files
    arrive one at a time). Kept separate from load_documents() above,
    which is built around DirectoryLoader for the bulk CLI ingestion
    path - different enough mechanics that unifying them would add
    complexity without removing any.
    """
    ext = Path(file_path).suffix.lower()
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    raise ValueError(
        f"Unsupported file extension '{ext}'. Supported: .txt, .md, .pdf"
    )


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


def get_embedding_model():
    """
    Local embedding model - ONNX runtime, CPU-only, no API key, no
    PyTorch dependency. First call downloads the model weights (~80MB)
    once and caches them.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_vectorstore(chunks, persist_dir: str):
    """
    Embed all chunks and persist them to a Chroma collection on disk.
    Running this again with the same persist_dir will add to the
    existing collection (fine for now; de-duplication comes later).
    """
    embeddings = get_embedding_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(persist_dir: str):
    """Reopen an existing Chroma collection for querying."""
    embeddings = get_embedding_model()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def ingest(docs_dir: str, persist_dir: str):
    """End-to-end: load -> chunk -> embed -> persist. Returns the store."""
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunk(s).")
    vectorstore = build_vectorstore(chunks, persist_dir)
    print(f"Vector store persisted to: {persist_dir}")
    return vectorstore
