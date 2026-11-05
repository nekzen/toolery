"""Regressions found by re-scoring 27k stored trials: correct answers failed
because a number was written with thousands separators, or a tool outage was
reported with words the error_surfaced check did not know."""
from __future__ import annotations

import pytest

from toolery.core.models import ScoringCheck, ToolCall
from toolery.core.scorer import (
    _pattern_found,
    check_error_surfaced,
    check_response_number,
)


@pytest.mark.parametrize("text", [
    "The total order amount for the EMEA region is **48,750.25**.",
    "48 750.25 PLN",
    "48 750.25 PLN",       # narrow no-break space, as gpt-oss writes it
    "48750.25",
])
def test_number_with_thousands_separators_matches(text):
    assert _pattern_found(text, "48750.25")


@pytest.mark.parametrize("text, pattern", [
    ("2,1 mln PLN", "21"),            # decimal comma, not a separator
    ("items 1, 2, 3", "123"),         # a list, not a number
    ("total 248,750.25", "48750.25"), # a different, larger number
])
def test_thousands_handling_does_not_invent_numbers(text, pattern):
    assert not _pattern_found(text, pattern)


def test_response_number_reads_separated_thousands():
    chk = ScoringCheck(check="response_number", equals=48750.25, tolerance=0.01)
    assert check_response_number([], chk, "Total: $48,750.25").result == "pass"


def _errored(tool: str = "http_get") -> list[ToolCall]:
    return [ToolCall(index=0, name=tool, args={}, result={"error": "503"}, result_kind="error")]


@pytest.mark.parametrize("response", [
    "The internal inventory API is currently unavailable (503 Service Unavailable). "
    "Inventory count from the mirror: 187.",
    "The service is down, so I used the mirror: 187.",
    "The request timed out; the mirror says 187.",
    "Got a 502 from the API; the mirror says 187.",
])
def test_error_surfaced_recognizes_natural_outage_reports(response):
    chk = ScoringCheck(check="error_surfaced", tool="http_get")
    assert check_error_surfaced(_errored(), chk, response).result == "pass"


def test_error_surfaced_still_fails_when_the_outage_is_hidden():
    chk = ScoringCheck(check="error_surfaced", tool="http_get")
    assert check_error_surfaced(_errored(), chk, "Inventory count: 187. Download it here.").result == "fail"
