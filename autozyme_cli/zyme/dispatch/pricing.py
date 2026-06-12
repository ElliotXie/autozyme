"""Model price registry for dispatch usage estimates.

Prices are USD per 1M tokens. Keep this table small, explicit, and sourced:
it is an estimate layer over agent telemetry, not an accounting ledger.
"""
from __future__ import annotations

import re
from typing import Any


PRICE_TABLE_UPDATED_AT = "2026-05-11"


MODEL_PRICES: list[dict[str, Any]] = [
    {
        "id": "cursor:composer-2-fast",
        "provider": "cursor",
        "model": "Composer 2 Fast",
        "aliases": ["composer-2-fast", "composer 2 fast", "composer2 fast"],
        "input_per_mtok": 1.50,
        "cached_input_per_mtok": None,
        "cache_write_per_mtok": 1.50,
        "output_per_mtok": 7.50,
        "source_url": "https://www.cursor.com/blog/composer-2",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": [
            "Cursor publishes input/output prices only; cache reads are estimated at input price."
        ],
    },
    {
        "id": "cursor:composer-2",
        "provider": "cursor",
        "model": "Composer 2",
        "aliases": ["composer-2", "composer 2", "composer2"],
        "input_per_mtok": 0.50,
        "cached_input_per_mtok": None,
        "cache_write_per_mtok": 0.50,
        "output_per_mtok": 2.50,
        "source_url": "https://www.cursor.com/blog/composer-2",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": [
            "Cursor publishes input/output prices only; cache reads are estimated at input price."
        ],
    },
    {
        "id": "openai:gpt-5.5",
        "provider": "openai",
        "model": "GPT-5.5",
        "aliases": ["gpt-5.5"],
        "input_per_mtok": 5.00,
        "cached_input_per_mtok": 0.50,
        "cache_write_per_mtok": 5.00,
        "output_per_mtok": 30.00,
        "source_url": "https://openai.com/api/pricing/",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": [],
    },
    {
        "id": "openai:gpt-5.4",
        "provider": "openai",
        "model": "GPT-5.4",
        "aliases": ["gpt-5.4"],
        "input_per_mtok": 2.50,
        "cached_input_per_mtok": 0.25,
        "cache_write_per_mtok": 2.50,
        "output_per_mtok": 15.00,
        "source_url": "https://openai.com/api/pricing/",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": [],
    },
    {
        "id": "openai:gpt-5.4-mini",
        "provider": "openai",
        "model": "GPT-5.4 mini",
        "aliases": ["gpt-5.4-mini", "gpt-5.4 mini", "gpt 5.4 mini"],
        "input_per_mtok": 0.75,
        "cached_input_per_mtok": 0.075,
        "cache_write_per_mtok": 0.75,
        "output_per_mtok": 4.50,
        "source_url": "https://openai.com/api/pricing/",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": [],
    },
    {
        "id": "openai:gpt-5.3-codex",
        "provider": "openai",
        "model": "GPT-5.3-Codex",
        "aliases": ["gpt-5.3-codex", "gpt 5.3 codex"],
        "input_per_mtok": 1.75,
        "cached_input_per_mtok": 0.175,
        "cache_write_per_mtok": 1.75,
        "output_per_mtok": 14.00,
        "source_url": "https://developers.openai.com/api/docs/models/gpt-5.3-codex",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": [],
    },
    {
        "id": "openai:gpt-5.2",
        "provider": "openai",
        "model": "GPT-5.2",
        "aliases": ["gpt-5.2", "gpt-5.2-codex", "gpt 5.2 codex"],
        "input_per_mtok": 1.75,
        "cached_input_per_mtok": 0.175,
        "cache_write_per_mtok": 1.75,
        "output_per_mtok": 14.00,
        "source_url": "https://openai.com/api/pricing/",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": [],
    },
    {
        "id": "anthropic:claude-opus-4-latest",
        "provider": "anthropic",
        "model": "Claude Opus 4.x",
        "aliases": [
            "claude-opus-4-8", "claude-opus-4.8",
            "claude-opus-4-8[1m]", "claude-opus-4.8[1m]",
            "claude-opus-4-7", "claude-opus-4.7",
            "claude-opus-4-7[1m]", "claude-opus-4.7[1m]",
            "claude-opus-4-6", "claude-opus-4.6",
            "claude-opus-4-6[1m]", "claude-opus-4.6[1m]",
            "claude-opus-4-5", "claude-opus-4.5",
        ],
        "input_per_mtok": 5.00,
        "cached_input_per_mtok": 0.50,
        "cache_write_per_mtok": 6.25,
        "output_per_mtok": 25.00,
        "source_url": "https://docs.claude.com/en/docs/about-claude/pricing",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": ["Cache write uses Anthropic 5-minute cache write pricing."],
    },
    {
        "id": "anthropic:claude-sonnet-4-latest",
        "provider": "anthropic",
        "model": "Claude Sonnet 4.x",
        "aliases": [
            "claude-sonnet-4-6", "claude-sonnet-4.6",
            "claude-sonnet-4-5", "claude-sonnet-4.5",
            "claude-sonnet-4", "claude-sonnet-4.0",
        ],
        "input_per_mtok": 3.00,
        "cached_input_per_mtok": 0.30,
        "cache_write_per_mtok": 3.75,
        "output_per_mtok": 15.00,
        "source_url": "https://docs.claude.com/en/docs/about-claude/pricing",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": ["Cache write uses Anthropic 5-minute cache write pricing."],
    },
    {
        "id": "anthropic:claude-haiku-4.5",
        "provider": "anthropic",
        "model": "Claude Haiku 4.5",
        "aliases": ["claude-haiku-4-5", "claude-haiku-4.5"],
        "input_per_mtok": 1.00,
        "cached_input_per_mtok": 0.10,
        "cache_write_per_mtok": 1.25,
        "output_per_mtok": 5.00,
        "source_url": "https://docs.claude.com/en/docs/about-claude/pricing",
        "source_checked_at": PRICE_TABLE_UPDATED_AT,
        "notes": ["Cache write uses Anthropic 5-minute cache write pricing."],
    },
]


def list_prices() -> list[dict[str, Any]]:
    """Return a serializable copy of the built-in price table."""
    return [dict(row) for row in MODEL_PRICES]


def find_model_price(model_name: str | None) -> dict[str, Any] | None:
    if not model_name:
        return None
    wanted = _norm(model_name)
    for row in MODEL_PRICES:
        keys = [row["id"], row["model"], *row.get("aliases", [])]
        if wanted in {_norm(x) for x in keys}:
            return dict(row)
    return None


def estimate_usage_cost(usage: dict[str, Any], model_name: str | None) -> dict[str, Any] | None:
    price = find_model_price(model_name)
    if price is None:
        return None

    input_tokens = _as_int(usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    cache_read_tokens = _as_int(usage.get("cache_read_tokens"))
    cache_write_tokens = _as_int(usage.get("cache_write_tokens"))

    input_usd = input_tokens * float(price["input_per_mtok"]) / 1_000_000
    output_usd = output_tokens * float(price["output_per_mtok"]) / 1_000_000

    cached_price = price.get("cached_input_per_mtok")
    cache_read_price = float(cached_price if cached_price is not None else price["input_per_mtok"])
    cache_read_usd = cache_read_tokens * cache_read_price / 1_000_000

    cache_write_price = float(price.get("cache_write_per_mtok") or price["input_per_mtok"])
    cache_write_usd = cache_write_tokens * cache_write_price / 1_000_000

    notes = list(price.get("notes") or [])
    if (
        cached_price is None and cache_read_tokens
        and not any("cache reads" in note.lower() for note in notes)
    ):
        notes.append("Cache read tokens estimated at input-token price.")

    total = input_usd + output_usd + cache_read_usd + cache_write_usd
    return {
        "model_price_id": price["id"],
        "model": price["model"],
        "source_url": price["source_url"],
        "source_checked_at": price["source_checked_at"],
        "input_usd": input_usd,
        "output_usd": output_usd,
        "cache_read_usd": cache_read_usd,
        "cache_write_usd": cache_write_usd,
        "total_usd": total,
        "notes": notes,
    }


def render_price_table() -> str:
    lines = [
        "# model prices",
        f"updated_at: {PRICE_TABLE_UPDATED_AT}",
        "",
        f"{'id':<36} {'input/M':>9} {'cached/M':>9} {'write/M':>9} {'output/M':>9} source",
    ]
    for row in MODEL_PRICES:
        cached = row.get("cached_input_per_mtok")
        cached_s = "input" if cached is None else _fmt_price(cached)
        lines.append(
            f"{row['id']:<36} "
            f"{_fmt_price(row['input_per_mtok']):>9} "
            f"{cached_s:>9} "
            f"{_fmt_price(row['cache_write_per_mtok']):>9} "
            f"{_fmt_price(row['output_per_mtok']):>9} "
            f"{row['source_url']}"
        )
    return "\n".join(lines)


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", str(value).strip().lower()).strip("-")


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _fmt_price(value: Any) -> str:
    return f"${float(value):g}"
