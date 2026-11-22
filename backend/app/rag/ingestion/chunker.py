"""Split documents into overlapping, heading-aware chunks.

Strategy, per document:
1. Split into sections at Markdown headings. Chunks never cross a section, so each
   chunk has one clear heading to cite (e.g. "Query Parameters › Optional parameters").
2. Split each section into blocks: paragraphs, list groups and whole ``` code fences.
3. Pack blocks into chunks of at most `size` characters. A block that is too big on
   its own is split by lines, then sentences, then hard-cut at a word boundary.
4. Start each new chunk with the tail of the previous one (`overlap` fraction of
   `size`) so a sentence cut at a boundary still appears whole in one chunk.
"""

import re
from collections.abc import Iterable, Iterator

from app.schemas.document import Chunk, Document

HEADING = re.compile(r"^(#{1,6}) +(.+?)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_documents(docs: Iterable[Document], size: int, overlap: float = 0.12) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for heading, section in _sections(doc.text):
            for text in _pack(_blocks(section), size, int(size * overlap)):
                chunks.append(
                    Chunk(id=len(chunks), text=text, source=doc.source, heading=heading, page=doc.page)
                )
    return chunks


def _sections(text: str) -> Iterator[tuple[str | None, str]]:
    """Yield (heading path, body) pairs. Heading path is 'H1 › H2 › H3' style."""
    path: list[tuple[int, str]] = []  # stack of (level, title)
    body: list[str] = []
    in_fence = False

    def current() -> str | None:
        return " › ".join(title for _, title in path) or None

    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            if "".join(body).strip():
                yield current(), "\n".join(body)
            body = []
            level = len(match.group(1))
            path = [(lvl, t) for lvl, t in path if lvl < level] + [(level, match.group(2))]
        else:
            body.append(line)
    if "".join(body).strip():
        yield current(), "\n".join(body)


def _blocks(section: str) -> list[str]:
    """Split on blank lines, but keep a code fence together even if it contains blank lines."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in section.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(blocks: list[str], size: int, overlap: int) -> Iterator[str]:
    pieces = [p for block in blocks for p in _fit(block, size)]
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        yield current
        tail = _tail(current, overlap)
        current = f"{tail}\n\n{piece}" if tail and len(tail) + 2 + len(piece) <= size else piece
    if current:
        yield current


def _fit(block: str, size: int) -> list[str]:
    """Break a single block into pieces no longer than size: lines → sentences → words."""
    if len(block) <= size:
        return [block]
    for splitter in (lambda b: b.split("\n"), SENTENCE_END.split):
        parts = [p for p in splitter(block) if p.strip()]
        if len(parts) > 1:
            return list(_merge(parts, size, sep="\n" if "\n" in block else " "))
    return _hard_cut(block, size)


def _merge(parts: list[str], size: int, sep: str) -> Iterator[str]:
    current = ""
    for part in parts:
        for p in _fit(part, size):
            candidate = f"{current}{sep}{p}" if current else p
            if len(candidate) <= size:
                current = candidate
            else:
                yield current
                current = p
    if current:
        yield current


def _hard_cut(text: str, size: int) -> list[str]:
    pieces = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return pieces + [text] if text else pieces


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap characters of text, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    start = text.find(" ", len(text) - overlap)
    return text[start + 1 :].strip() if start != -1 else ""
