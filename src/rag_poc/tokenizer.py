"""Tokenização para o índice lexical (BM25), otimizada para inglês."""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

# Stopwords clássicas do inglês (inclui contrações comuns).
STOPWORDS_EN = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "ain",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "couldn't",
        "d",
        "did",
        "didn",
        "didn't",
        "do",
        "does",
        "doesn",
        "doesn't",
        "doing",
        "don",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "hadn't",
        "has",
        "hasn",
        "hasn't",
        "have",
        "haven",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "mightn",
        "mightn't",
        "more",
        "most",
        "mustn",
        "mustn't",
        "my",
        "myself",
        "needn",
        "needn't",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "shan't",
        "she",
        "she's",
        "should",
        "should've",
        "shouldn",
        "shouldn't",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "that'll",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "wasn't",
        "we",
        "were",
        "weren",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "y",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = snowballstemmer.stemmer("english")


def strip_accents(text: str) -> str:
    """Remove acentos preservando o restante do texto (ex.: café -> cafe)."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def stem(token: str) -> str:
    """Stemming Snowball (inglês). Números são mantidos intactos."""
    if token.isdigit():
        return token
    return _STEMMER.stemWord(token)


def tokenize_en(text: str) -> list[str]:
    """Normaliza o texto e retorna tokens úteis para busca lexical em inglês.

    - minúsculas e sem acentos;
    - apenas blocos alfanuméricos;
    - stopwords removidas;
    - tokens de 1 caractere descartados, exceto dígitos;
    - stemming Snowball (``components`` -> ``compon``, ``rendering`` -> ``render``).
    """
    if not text:
        return []
    normalized = strip_accents(text.lower())
    return [
        stem(token)
        for token in _TOKEN_RE.findall(normalized)
        if token not in STOPWORDS_EN and (len(token) > 1 or token.isdigit())
    ]
