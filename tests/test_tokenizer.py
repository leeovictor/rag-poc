from __future__ import annotations

from rag_poc.tokenizer import strip_accents, tokenize_en


def test_strip_accents() -> None:
    assert strip_accents("Café résumé") == "Cafe resume"


def test_stopwords_are_removed() -> None:
    assert tokenize_en("The components are rendering") == ["compon", "render"]


def test_stemming_unifies_plural_and_gerund() -> None:
    assert tokenize_en("component components rendering rendered") == [
        "compon",
        "compon",
        "render",
        "render",
    ]


def test_accents_are_normalized() -> None:
    assert "cafe" in tokenize_en("café")


def test_numbers_are_kept() -> None:
    assert "2024" in tokenize_en("React 2024")


def test_single_letters_are_discarded_except_digits() -> None:
    assert tokenize_en("x y 1") == ["1"]


def test_empty_text() -> None:
    assert tokenize_en("") == []
