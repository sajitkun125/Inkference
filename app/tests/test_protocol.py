"""Parser tests for the agent's JSON action protocol.

This is the highest-risk component in the agent: a small model's output is the only
thing standing between the graph and a crash, and every malformed case below was
either observed from a real provider or is a near-miss worth pinning. No network.
"""
from __future__ import annotations

import pytest

from inkference.agent.protocol import TERMINAL, parse_action, render_tool_catalog


def action_of(text: str) -> dict | None:
    return parse_action(text)[0]


# --------------------------------------------------------------------------- #
# well-formed
# --------------------------------------------------------------------------- #
def test_plain_object():
    assert action_of('{"action":"search","query":"Plymouth"}') == {
        "action": "search", "query": "Plymouth",
    }


def test_fenced_json_block():
    text = '```json\n{"action": "read_range", "start": 118, "end": 123}\n```'
    assert action_of(text) == {"action": "read_range", "start": 118, "end": 123}


def test_object_wrapped_in_prose():
    text = 'Sure! Here it is: {"action":"read_page","page":42} Hope that helps.'
    assert action_of(text) == {"action": "read_page", "page": 42}


def test_braces_inside_a_string_do_not_confuse_the_extractor():
    assert action_of('{"action":"search","query":"a {nested} brace"}') == {
        "action": "search", "query": "a {nested} brace",
    }


def test_escaped_quotes_survive():
    assert action_of(r'{"action":"search","query":"say \"hi\" now"}') == {
        "action": "search", "query": 'say "hi" now',
    }


# --------------------------------------------------------------------------- #
# repairs (each of these was emitted by a real provider)
# --------------------------------------------------------------------------- #
def test_trailing_comma():
    assert action_of('{"action":"search","query":"x","k":6,}') == {
        "action": "search", "query": "x", "k": 6,
    }


def test_stray_quote_after_a_number():
    # Observed from groq openai/gpt-oss-120b: {"...,"end":26"}
    text = '{"standalone_question":"q","action":"read_range","start":20,"end":26"}'
    assert action_of(text) == {"action": "read_range", "start": 20, "end": 26}


def test_repairs_do_not_corrupt_a_quoted_number():
    # The stray-quote repair must not eat a legitimately quoted numeric string.
    assert action_of('{"action":"search","query":"12345"}') == {
        "action": "search", "query": "12345",
    }


# --------------------------------------------------------------------------- #
# coercion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ('{"action":"read_page","page":"42"}', 42),
    ('{"action":"read_page","page":"page 42"}', 42),
    ('{"action":"read_page","page":42.0}', 42),
])
def test_int_args_are_coerced(raw, expected):
    assert action_of(raw)["page"] == expected


def test_action_name_is_case_and_space_insensitive():
    assert action_of('{"action":"  SEARCH ","query":"x"}')["action"] == "search"


# --------------------------------------------------------------------------- #
# graceful degradation — none of these may raise
# --------------------------------------------------------------------------- #
def test_unknown_action_degrades_to_answer():
    assert action_of('{"action":"fly_to_moon"}')["action"] == TERMINAL


def test_missing_required_arg_degrades_to_answer():
    assert action_of('{"action":"search"}')["action"] == TERMINAL


@pytest.mark.parametrize("text", ["", "I think we should search the journal.", "   ", "[1,2,3]"])
def test_no_json_returns_none_so_caller_can_retry(text):
    # None is the signal for "spend one repair retry" — distinct from a degraded action.
    assert action_of(text) is None


def test_unparseable_never_raises():
    for junk in ('{"action":', '{{{{', '}{', '{"a":1'):
        parse_action(junk)  # must not raise


# --------------------------------------------------------------------------- #
# extras
# --------------------------------------------------------------------------- #
def test_standalone_question_is_extracted_alongside_the_action():
    text = ('{"standalone_question":"What was the weather at Plymouth?",'
            '"action":"search","query":"Plymouth weather","k":6}')
    action, extras = parse_action(text)
    assert action == {"action": "search", "query": "Plymouth weather", "k": 6}
    assert extras["standalone_question"] == "What was the weather at Plymouth?"


def test_blank_standalone_question_is_dropped():
    _action, extras = parse_action('{"standalone_question":"  ","action":"overview"}')
    assert "standalone_question" not in (extras or {})


def test_catalog_lists_every_action():
    catalog = render_tool_catalog()
    for name in ("search", "read_page", "read_range", "overview", "answer"):
        assert name in catalog
