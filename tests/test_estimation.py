from __future__ import annotations

import pytest

from rag_poc import estimation


def test_embedding_cost_for_known_model() -> None:
    cost = estimation.estimate_embedding_cost_usd("openai/text-embedding-3-small", 1_000_000)
    assert cost == pytest.approx(0.02)


def test_embedding_cost_falls_back_to_default_price() -> None:
    cost = estimation.estimate_embedding_cost_usd("custom/model", 1_000_000)
    assert cost == pytest.approx(estimation.DEFAULT_EMBEDDING_PRICE_USD_PER_MTOK)


def test_embedding_time_range_scales_with_requests() -> None:
    low, high = estimation.estimate_embedding_time_seconds(chunks=640, batch_size=64)
    assert low == pytest.approx(10 * 1.2)
    assert high == pytest.approx(10 * 3.0)


def test_embedding_time_range_is_zero_without_chunks() -> None:
    assert estimation.estimate_embedding_time_seconds(0, 64) == (0.0, 0.0)


def test_chat_cost_combines_input_and_output() -> None:
    cost = estimation.estimate_chat_cost_usd(
        "openai/gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == pytest.approx(0.75)


def test_count_tokens_returns_positive_value() -> None:
    tokens, _exact = estimation.count_tokens(["hello world", "react components"])
    assert tokens > 0
