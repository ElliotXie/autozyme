from zyme.cli import build_parser
from zyme.prompts import sync_prompts


def _parser_parts():
    parser = build_parser()
    return parser, sync_prompts.all_subcommands(parser)


def test_validate_invocation_uses_nested_subcommand_flags():
    parser, subparsers = _parser_parts()

    ok, reason = sync_prompts.validate_invocation(
        "baseline",
        "record --tier tiny --speed-sec <secs> --peak-mb <mb>",
        parser,
        subparsers,
    )

    assert ok, reason


def test_validate_invocation_handles_markdown_optional_flags():
    parser, subparsers = _parser_parts()

    ok, reason = sync_prompts.validate_invocation(
        "baseline",
        "record --tier <tier> [--speed-sec <s> --peak-mb <m>] [--oom] [--force]",
        parser,
        subparsers,
    )

    assert ok, reason


def test_validate_invocation_still_reports_unknown_nested_flags():
    parser, subparsers = _parser_parts()

    ok, reason = sync_prompts.validate_invocation(
        "baseline",
        "record --tier tiny --not-a-real-flag",
        parser,
        subparsers,
    )

    assert not ok
    assert "baseline record" in reason
    assert "--not-a-real-flag" in reason
