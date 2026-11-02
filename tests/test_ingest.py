"""
The loaders, which are now local functions rather than
langchain_community's DirectoryLoader / TextLoader / PyPDFLoader.

Worth testing properly because ingestion failures are silent: a file
that loads as empty text produces chunks that embed to nothing useful,
and nothing downstream reports that the document is effectively absent
from the knowledge base.
"""
import pytest

from src.ingest import chunk_documents, load_documents, load_file


def test_loads_a_text_file(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("some content")
    docs = load_file(path)
    assert len(docs) == 1
    assert docs[0].page_content == "some content"
    assert docs[0].metadata["source"] == str(path)


def test_loads_a_markdown_file(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# Heading\n\nBody.")
    assert "Heading" in load_file(path)[0].page_content


def test_an_empty_file_yields_no_documents(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ")
    assert load_file(path) == []


def test_rejects_an_unsupported_extension(tmp_path):
    path = tmp_path / "a.exe"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_file(path)


def test_survives_invalid_utf8(tmp_path):
    # One bad byte in a long export shouldn't abort the whole run.
    path = tmp_path / "a.txt"
    path.write_bytes(b"valid text \xff\xfe more text")
    assert "valid text" in load_file(path)[0].page_content


def test_load_documents_walks_subdirectories(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "top.txt").write_text("top level")
    (tmp_path / "nested" / "deep.md").write_text("nested level")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 2


def test_load_documents_ignores_unsupported_files(tmp_path):
    (tmp_path / "keep.txt").write_text("keep me")
    (tmp_path / "skip.json").write_text("{}")
    docs = load_documents(str(tmp_path))
    assert len(docs) == 1


def test_load_documents_is_order_stable(tmp_path):
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).write_text(name)
    first = [d.page_content for d in load_documents(str(tmp_path))]
    assert first == sorted(first)


def test_load_documents_raises_on_a_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_documents("/no/such/directory")


def test_load_documents_raises_when_nothing_is_readable(tmp_path):
    (tmp_path / "skip.json").write_text("{}")
    with pytest.raises(ValueError, match="Nothing to ingest"):
        load_documents(str(tmp_path))


def test_chunking_overlaps_so_boundary_facts_survive(tmp_path):
    from src.config import config

    long_text = " ".join(f"sentence number {i}." for i in range(400))
    path = tmp_path / "long.txt"
    path.write_text(long_text)

    chunks = chunk_documents(load_file(path))
    assert len(chunks) > 1
    assert all(len(c.page_content) <= config.CHUNK_SIZE for c in chunks)
    # Consecutive chunks share text, so a fact on a boundary isn't
    # severed from its surroundings.
    assert chunks[0].page_content[-20:] in chunks[1].page_content


def test_chunk_metadata_is_preserved(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("x " * 2000)
    for chunk in chunk_documents(load_file(path)):
        assert chunk.metadata["source"] == str(path)
