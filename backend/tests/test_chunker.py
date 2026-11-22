from app.rag.ingestion.chunker import chunk_documents
from app.schemas.document import Document

DOC = """# Query Parameters

Intro paragraph about query parameters.

## Defaults

They can have default values like skip=0.

```python
def read(skip: int = 0):

    return skip
```

## Optional parameters

### Details

Set the default to None to make it optional.
"""


def chunk(text: str, size: int = 2000, overlap: float = 0.0):
    return chunk_documents([Document(source="q.md", text=text)], size=size, overlap=overlap)


def test_headings_become_paths():
    headings = [c.heading for c in chunk(DOC)]
    assert headings == [
        "Query Parameters",
        "Query Parameters › Defaults",
        "Query Parameters › Optional parameters › Details",
    ]


def test_code_fence_is_kept_whole():
    defaults = chunk(DOC)[1].text
    assert "def read(skip: int = 0):\n\n    return skip\n```" in defaults


def test_heading_inside_code_is_not_a_heading():
    text = "# Real\n\n```bash\n# a shell comment\necho hi\n```"
    chunks = chunk(text)
    assert len(chunks) == 1 and chunks[0].heading == "Real"


def test_chunks_respect_size_and_cover_all_text():
    words = [f"word{i}." for i in range(400)]
    text = "# T\n\n" + "\n\n".join(" ".join(words[i : i + 20]) for i in range(0, 400, 20))
    chunks = chunk(text, size=300)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert all(w in joined for w in words)


def test_long_paragraph_is_split():
    text = "# T\n\n" + " ".join(f"Sentence number {i}." for i in range(200))
    chunks = chunk(text, size=300)
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[-1].text.endswith("Sentence number 199.")


def test_overlap_repeats_tail_of_previous_chunk():
    text = "# T\n\n" + "\n\n".join(f"Paragraph {i} " + "filler " * 15 for i in range(10))
    chunks = chunk(text, size=300, overlap=0.2)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.text.split()[-1] in nxt.text.split()[:15]


def test_ids_are_sequential_across_documents():
    docs = [Document(source=s, text=DOC) for s in ("a.md", "b.md")]
    chunks = chunk_documents(docs, size=2000, overlap=0.0)
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert {c.source for c in chunks} == {"a.md", "b.md"}
