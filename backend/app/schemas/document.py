from pydantic import BaseModel


class Document(BaseModel):
    """One loaded file, or one page of a PDF."""

    source: str  # path relative to the corpus folder, e.g. "tutorial/query-params.md"
    text: str
    page: int | None = None


class Chunk(BaseModel):
    id: int
    text: str
    source: str
    heading: str | None = None
    page: int | None = None


class RetrievedChunk(Chunk):
    score: float


class Citation(BaseModel):
    n: int  # the [n] marker used in the answer
    source: str
    heading: str | None = None
    page: int | None = None
