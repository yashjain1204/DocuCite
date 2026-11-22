"""Read files from disk into Document objects."""

from pathlib import Path

from pypdf import PdfReader

from app.rag.ingestion.cleaners import clean_markdown
from app.schemas.document import Document

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def load_markdown(path: Path, root: Path) -> list[Document]:
    text = clean_markdown(path.read_text(encoding="utf-8"))
    return [Document(source=_relative(path, root), text=text)]


def load_text(path: Path, root: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(source=_relative(path, root), text=text)]


def load_pdf(path: Path, root: Path) -> list[Document]:
    reader = PdfReader(path)
    source = _relative(path, root)
    return [
        Document(source=source, text=text, page=i)
        for i, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]


LOADERS = {".md": load_markdown, ".txt": load_text, ".pdf": load_pdf}


def load_file(path: Path, root: Path) -> list[Document]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.name}")
    return loader(path, root)


def load_directory(root: Path) -> list[Document]:
    """Load every supported file under root, skipping files that end up empty."""
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(d for d in load_file(path, root) if d.text)
    return docs


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
