from __future__ import annotations

import pytest

from rag_poc.errors import PathNotFoundError, UnsupportedFileError
from rag_poc.loaders import discover_files, load_documents, load_file


def test_load_txt_file(tmp_path) -> None:
    path = tmp_path / "nota.txt"
    path.write_text("Hello world.\nSecond line.", encoding="utf-8")

    documents = load_file(path)

    assert len(documents) == 1
    assert documents[0].metadata["file_name"] == "nota.txt"
    assert "page" not in documents[0].metadata
    assert "Hello world" in documents[0].page_content


def test_load_txt_latin1(tmp_path) -> None:
    path = tmp_path / "latin.txt"
    path.write_bytes("Ação e coração".encode("latin-1"))

    documents = load_file(path)

    assert "Ação e coração" in documents[0].page_content


def test_load_markdown_cleans_structure(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        """---
title: Describing the UI
---

import {Intro} from './Intro';

<Intro>

## Hello {/*hello*/}

```js
import { useState } from 'react';
```

{/* hidden comment */}
""",
        encoding="utf-8",
    )

    documents = load_file(path)
    text = documents[0].page_content

    assert "title:" not in text
    assert "import {Intro}" not in text
    assert "import { useState } from 'react';" in text
    assert "{/* hidden comment */}" not in text
    assert "## Hello" in text


def test_empty_text_file_raises(tmp_path) -> None:
    path = tmp_path / "vazio.txt"
    path.write_text("   \n\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_file(path)


def test_discover_directory_ignores_unsupported_and_hidden(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "c.mdx").write_text("c", encoding="utf-8")
    (tmp_path / "d.md").write_text("d", encoding="utf-8")
    (tmp_path / "e.html").write_text("e", encoding="utf-8")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "f.txt").write_text("f", encoding="utf-8")

    files = discover_files(tmp_path)

    assert [file.name for file in files] == ["a.txt", "b.md", "c.mdx", "d.md"]


def test_discover_unsupported_file_raises(tmp_path) -> None:
    path = tmp_path / "x.html"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(UnsupportedFileError):
        discover_files(path)


def test_discover_missing_path_raises(tmp_path) -> None:
    with pytest.raises(PathNotFoundError):
        discover_files(tmp_path / "nao-existe")


def test_load_documents_accumulates_errors(tmp_path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("conteúdo válido", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("   ", encoding="utf-8")

    report = load_documents([good, bad])

    assert len(report.documents) == 1
    assert report.files == [good]
    assert len(report.errors) == 1
    assert report.errors[0].path == bad
