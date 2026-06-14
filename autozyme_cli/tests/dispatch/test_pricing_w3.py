"""Wave-3 coverage for zyme.dispatch.pricing.

tests/dispatch/test_usage.py drives find_model_price + estimate_usage_cost
end-to-end through collect_usage, but never touches list_prices(),
render_price_table(), the unknown-model None fallbacks, the cache-read
estimation note, or the _as_int / _fmt_price edge helpers directly. This
file fills those reachable gaps (pricing is pure-python, no subprocess).
"""
from __future__ import annotations

import pytest

from zyme.dispatch import pricing
from zyme.dispatch.pricing import (
    MODEL_PRICES,
    PRICE_TABLE_UPDATED_AT,
    _as_int,
    _fmt_price,
    _norm,
    estimate_usage_cost,
    find_model_price,
    list_prices,
    render_price_table,
)


# --------------------------------------------------------------------------
# list_prices
# --------------------------------------------------------------------------

class TestListPrices:
    def test_returns_one_row_per_model(self):
        rows = list_prices()
        assert len(rows) == len(MODEL_PRICES)

    def test_rows_are_copies_not_aliases(self):
        rows = list_prices()
        # mutating the returned copy must not corrupt the registry.
        rows[0]["input_per_mtok"] = -999
        assert MODEL_PRICES[0]["input_per_mtok"] != -999

    def test_each_row_has_required_keys(self):
        for row in list_prices():
            for key in ("id", "provider", "model", "input_per_mtok",
                        "output_per_mtok", "source_url"):
                assert key in row, f"{row.get('id')} missing {key}"


# --------------------------------------------------------------------------
# find_model_price — alias resolution + None fallbacks
# --------------------------------------------------------------------------

class TestFindModelPrice:
    def test_none_model_name(self):
        assert find_model_price(None) is None

    def test_empty_string(self):
        assert find_model_price("") is None

    def test_resolves_by_canonical_id(self):
        row = find_model_price("anthropic:claude-opus-4-latest")
        assert row is not None
        assert row["id"] == "anthropic:claude-opus-4-latest"

    def test_resolves_by_human_model_name(self):
        row = find_model_price("Claude Haiku 4.5")
        assert row is not None
        assert row["id"] == "anthropic:claude-haiku-4.5"

    @pytest.mark.parametrize("alias,expected_id", [
        ("claude-opus-4-8", "anthropic:claude-opus-4-latest"),
        ("claude-opus-4.8[1m]", "anthropic:claude-opus-4-latest"),
        ("claude-sonnet-4-6", "anthropic:claude-sonnet-4-latest"),
        ("gpt-5.5", "openai:gpt-5.5"),
        ("gpt-5.3-codex", "openai:gpt-5.3-codex"),
        ("composer-2-fast", "cursor:composer-2-fast"),
        ("composer2", "cursor:composer-2"),
    ])
    def test_resolves_by_alias(self, alias, expected_id):
        row = find_model_price(alias)
        assert row is not None
        assert row["id"] == expected_id

    def test_alias_is_case_and_separator_insensitive(self):
        # _norm lowercases and collapses non-alnum runs to '-'.
        assert find_model_price("GPT 5.5")["id"] == "openai:gpt-5.5"
        assert find_model_price("Composer 2 Fast")["id"] == "cursor:composer-2-fast"

    def test_unknown_model_returns_none(self):
        assert find_model_price("totally-made-up-model-9000") is None

    def test_returns_a_copy(self):
        row = find_model_price("gpt-5.5")
        row["input_per_mtok"] = -1
        assert find_model_price("gpt-5.5")["input_per_mtok"] != -1


# --------------------------------------------------------------------------
# estimate_usage_cost — cost math, unknown-model fallback, cache estimation
# --------------------------------------------------------------------------

class TestEstimateUsageCost:
    def test_unknown_model_returns_none(self):
        assert estimate_usage_cost({"input_tokens": 100}, "no-such-model") is None
        assert estimate_usage_cost({"input_tokens": 100}, None) is None

    def test_basic_input_output_math(self):
        # gpt-5.5: input 5.00/M, output 30.00/M.
        out = estimate_usage_cost(
            {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
            "gpt-5.5",
        )
        assert out["input_usd"] == pytest.approx(5.00)
        assert out["output_usd"] == pytest.approx(30.00)
        assert out["cache_read_usd"] == 0.0
        assert out["cache_write_usd"] == 0.0
        assert out["total_usd"] == pytest.approx(35.00)
        assert out["model_price_id"] == "openai:gpt-5.5"
        assert out["source_checked_at"] == PRICE_TABLE_UPDATED_AT

    def test_cache_read_uses_cached_price_when_present(self):
        # gpt-5.5: cached_input 0.50/M.
        out = estimate_usage_cost(
            {"cache_read_tokens": 1_000_000}, "gpt-5.5")
        assert out["cache_read_usd"] == pytest.approx(0.50)
        # cached price present -> no estimation note appended.
        assert not any("estimated at input" in n.lower() for n in out["notes"])

    def test_cache_write_math(self):
        # claude-opus: cache_write 6.25/M.
        out = estimate_usage_cost(
            {"cache_write_tokens": 1_000_000}, "anthropic:claude-opus-4-latest")
        assert out["cache_write_usd"] == pytest.approx(6.25)

    def test_cache_read_falls_back_to_input_price_when_cached_none(self):
        # cursor:composer-2 has cached_input_per_mtok=None -> falls back to
        # input price (0.50/M) AND appends the estimation note (line 206).
        out = estimate_usage_cost(
            {"cache_read_tokens": 1_000_000}, "composer-2")
        assert out["cache_read_usd"] == pytest.approx(0.50)
        # composer already documents cache reads in its notes, so the auto
        # note is NOT appended (the "cache reads" guard suppresses it).
        assert any("cache reads" in n.lower() for n in out["notes"])

    def test_cache_read_estimation_note_appended_when_undocumented(self, monkeypatch):
        # Craft a price row with cached=None and notes that do NOT mention
        # "cache reads", so the estimation note path (line 202-206) fires.
        fake = {
            "id": "fake:model", "model": "Fake", "aliases": ["fakey"],
            "input_per_mtok": 2.0, "cached_input_per_mtok": None,
            "cache_write_per_mtok": 2.0, "output_per_mtok": 4.0,
            "source_url": "http://x", "source_checked_at": "2026-01-01",
            "notes": [],
        }
        monkeypatch.setattr(pricing, "MODEL_PRICES", [fake])
        out = estimate_usage_cost({"cache_read_tokens": 1_000_000}, "fakey")
        assert any("Cache read tokens estimated at input-token price" in n
                   for n in out["notes"])

    def test_non_int_tokens_coerced_to_zero(self):
        out = estimate_usage_cost(
            {"input_tokens": "garbage", "output_tokens": None}, "gpt-5.5")
        assert out["input_usd"] == 0.0
        assert out["output_usd"] == 0.0
        assert out["total_usd"] == 0.0

    def test_notes_are_a_fresh_list(self):
        # mutating the returned notes must not poison the registry row's notes.
        before = len(find_model_price("composer-2")["notes"])
        out = estimate_usage_cost({"input_tokens": 1}, "composer-2")
        out["notes"].append("scratch")
        assert len(find_model_price("composer-2")["notes"]) == before


# --------------------------------------------------------------------------
# render_price_table
# --------------------------------------------------------------------------

class TestRenderPriceTable:
    def test_header_and_updated_at(self):
        table = render_price_table()
        assert table.startswith("# model prices")
        assert f"updated_at: {PRICE_TABLE_UPDATED_AT}" in table

    def test_one_line_per_model(self):
        table = render_price_table().splitlines()
        # header(4 lines) + one row per model.
        body = [ln for ln in table if ln.startswith(("cursor:", "openai:", "anthropic:"))]
        assert len(body) == len(MODEL_PRICES)

    def test_cached_none_rendered_as_literal_input(self):
        # cursor rows have cached_input_per_mtok=None -> column shows "input".
        table = render_price_table()
        cursor_line = next(ln for ln in table.splitlines()
                           if ln.startswith("cursor:composer-2 "))
        assert "input" in cursor_line

    def test_numeric_cached_rendered_as_price(self):
        table = render_price_table()
        opus_line = next(ln for ln in table.splitlines()
                         if ln.startswith("anthropic:claude-opus"))
        # cached 0.50 -> "$0.5"
        assert "$0.5" in opus_line


# --------------------------------------------------------------------------
# pure helpers: _as_int / _fmt_price / _norm
# --------------------------------------------------------------------------

class TestAsInt:
    def test_none(self):
        assert _as_int(None) == 0

    def test_bool_is_zero(self):
        # bool is an int subclass but is explicitly excluded.
        assert _as_int(True) == 0
        assert _as_int(False) == 0

    def test_int_passthrough(self):
        assert _as_int(42) == 42

    def test_float_truncated(self):
        assert _as_int(3.9) == 3

    def test_numeric_string(self):
        assert _as_int("17") == 17

    def test_garbage_string_is_zero(self):
        assert _as_int("not a number") == 0

    def test_list_is_zero(self):
        assert _as_int([1, 2]) == 0


class TestFmtPrice:
    def test_integer_value(self):
        assert _fmt_price(5) == "$5"

    def test_fractional_g_format(self):
        assert _fmt_price(0.50) == "$0.5"

    def test_string_numeric(self):
        assert _fmt_price("2.5") == "$2.5"


class TestNorm:
    def test_lowercases_and_dashes(self):
        assert _norm("GPT 5.5") == "gpt-5.5"

    def test_strips_leading_trailing_dashes(self):
        assert _norm("  Composer 2 Fast  ") == "composer-2-fast"

    def test_preserves_dots(self):
        assert _norm("claude-opus-4.8") == "claude-opus-4.8"
