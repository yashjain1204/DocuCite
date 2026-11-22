"""Turn MkDocs-flavoured Markdown (as used by the FastAPI docs) into plain, readable text."""

import re

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# [ \t]* rather than \s* at line ends, so the newline itself is kept
CODE_INCLUDE = re.compile(r"^\{\*.*\*\}[ \t]*$", re.MULTILINE)  # {* ../../docs_src/x.py hl[9] *}
HEADING_ANCHOR = re.compile(r"^(#{1,6} .*?)[ \t]*\{[ \t]*#[\w-]+[ \t]*\}[ \t]*$", re.MULTILINE)  # ## Title { #title }
ADMONITION_OPEN = re.compile(r"^[ \t]*/{3,4}[ \t]*(\w+)(?:[ \t]*\|[ \t]*(.*?))?[ \t]*$", re.MULTILINE)  # /// tip | Title
ADMONITION_CLOSE = re.compile(r"^[ \t]*/{3,4}[ \t]*$", re.MULTILINE)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
BLANK_LINES = re.compile(r"\n{3,}")


def _admonition(match: re.Match) -> str:
    kind, title = match.group(1), match.group(2)
    if kind == "tab":  # "//// tab | Python 3.10+" only switches code samples
        return ""
    label = kind.capitalize()
    return f"{label}: {title.strip()}" if title else f"{label}:"


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = FRONT_MATTER.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = CODE_INCLUDE.sub("", text)
    text = HEADING_ANCHOR.sub(r"\1", text)
    text = ADMONITION_CLOSE.sub("", text)
    text = ADMONITION_OPEN.sub(_admonition, text)
    text = HTML_TAG.sub("", text)
    text = BLANK_LINES.sub("\n\n", text)
    return text.strip()
