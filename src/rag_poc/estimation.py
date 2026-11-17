"""Estimativas de custo e tempo para ingestão e chat (sem chamar a API)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

# Preços em USD por 1M de tokens (referência OpenAI; o OpenRouter repassa
# esses valores para os modelos da OpenAI). Ajuste aqui se necessário.
EMBEDDING_PRICES_USD_PER_MTOK: dict[str, float] = {
    "openai/text-embedding-3-small": 0.02,
    "openai/text-embedding-3-large": 0.13,
    "openai/text-embedding-ada-002": 0.10,
}
DEFAULT_EMBEDDING_PRICE_USD_PER_MTOK = 0.02

# (entrada, saída) em USD por 1M de tokens.
CHAT_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1": (2.00, 8.00),
}
DEFAULT_CHAT_PRICES_USD_PER_MTOK = (0.15, 0.60)

# Latência observada típica por lote de embeddings (segundos), usada como faixa.
SECONDS_PER_EMBEDDING_REQUEST = (1.2, 3.0)

# Overhead aproximado do prompt (system + pergunta) em tokens.
PROMPT_OVERHEAD_TOKENS = 350
# Tamanho médio da resposta em tokens.
ANSWER_TOKENS = 250


@lru_cache(maxsize=1)
def _encoding() -> object:
    import tiktoken

    # cl100k_base é o encoding dos modelos text-embedding-3-*.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(texts: Sequence[str]) -> tuple[int, bool]:
    """Conta tokens com tiktoken; fallback heurístico (chars/4) se indisponível.

    Retorna ``(tokens, exato)``.
    """
    try:
        encoding = _encoding()
        total = sum(
            len(encoding.encode(text, disallowed_special=()))  # type: ignore[attr-defined]
            for text in texts
        )
        return total, True
    except Exception:  # noqa: BLE001 - tiktoken pode exigir download do vocabulário
        heuristic = sum(max(1, len(text) // 4) for text in texts)
        return heuristic, False


def embedding_price_per_mtok(model: str) -> float:
    return EMBEDDING_PRICES_USD_PER_MTOK.get(model, DEFAULT_EMBEDDING_PRICE_USD_PER_MTOK)


def estimate_embedding_cost_usd(model: str, tokens: int) -> float:
    return tokens / 1_000_000 * embedding_price_per_mtok(model)


def estimate_embedding_time_seconds(chunks: int, batch_size: int) -> tuple[float, float]:
    """Faixa (mínimo, máximo) de tempo apenas das chamadas de embedding."""
    if chunks <= 0 or batch_size <= 0:
        return 0.0, 0.0
    requests = math.ceil(chunks / batch_size)
    low, high = SECONDS_PER_EMBEDDING_REQUEST
    return requests * low, requests * high


def estimate_chat_cost_usd(
    model: str, input_tokens: int, output_tokens: int = ANSWER_TOKENS
) -> float:
    input_price, output_price = CHAT_PRICES_USD_PER_MTOK.get(
        model, DEFAULT_CHAT_PRICES_USD_PER_MTOK
    )
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
