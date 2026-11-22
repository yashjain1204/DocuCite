from pathlib import Path

from app.rag.ingestion.cleaners import clean_markdown
from app.rag.ingestion.loaders import load_directory

RAW = """---
title: x
---
# Query Parameters { #query-parameters }

<!-- hidden -->
Some <dfn title="converting">"parsing"</dfn> text.

{* ../../docs_src/query_params/tutorial001.py hl[9] *}

/// tip | Good to know

Use `None` as a default.

///

//// tab | Python 3.10+

```python
x = 1
```

////
"""


def test_clean_markdown():
    out = clean_markdown(RAW)
    assert out.startswith("# Query Parameters\n")
    assert "title: x" not in out
    assert "hidden" not in out
    assert "{*" not in out
    assert "<dfn" not in out and '"parsing"' in out
    assert "Tip: Good to know" in out
    assert "///" not in out and "tab |" not in out
    assert "x = 1" in out
    assert "\n\n\n" not in out


def test_load_directory(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("# A { #a }\n\nHello", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain", encoding="utf-8")
    (tmp_path / "empty.md").write_text("{* x.py *}", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    docs = load_directory(tmp_path)

    assert [d.source for d in docs] == ["b.txt", "sub/a.md"]
    assert docs[1].text == "# A\n\nHello"
