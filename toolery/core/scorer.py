from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import unicodedata
from collections.abc import Callable

import jsonschema
import yaml

from toolery.core.models import TURN_SEPARATOR, CheckResult, ScoringCheck, ToolCall
from toolery.core.text_utils import unwrap_structured_payload


def _implicit_partial_gradient_enabled() -> bool:
    """Read env at call time so tests / runs can toggle freely.

    Default OFF: any failed required (when forbidden clean) → status="fail",
    score = weights["fail"] (binary). Set TOOLERY_PARTIAL_GRADIENT=on to
    restore the previous behavior where partial credit was awarded
    proportional to how many required checks passed.
    """
    return os.environ.get("TOOLERY_PARTIAL_GRADIENT", "off").strip().lower() in {"1", "on", "true", "yes"}

CheckFn = Callable[[list[ToolCall], ScoringCheck, str | None], CheckResult]


def _ok(check: str, detail: str = "") -> CheckResult:
    return CheckResult(check=check, result="pass", detail=detail)


def _bad(check: str, detail: str = "") -> CheckResult:
    return CheckResult(check=check, result="fail", detail=detail)


def check_tool_called(calls, chk, response):
    target = chk.model_dump()["tool"]
    if any(c.name == target for c in calls):
        return _ok("tool_called", f"{target} was called")
    return _bad("tool_called", f"{target} was not called")


def check_tool_not_called(calls, chk, response):
    target = chk.model_dump()["tool"]
    if any(c.name == target for c in calls):
        return _bad("tool_not_called", f"{target} was called (forbidden)")
    return _ok("tool_not_called", f"{target} correctly not called")


def check_tool_args_contain(calls, chk, response):
    d = chk.model_dump()
    target = d["tool"]
    expected = d.get("args", {})
    for c in calls:
        if c.name != target:
            continue
        if all(c.args.get(k) == v for k, v in expected.items()):
            return _ok("tool_args_contain", f"{target} args matched {expected}")
    return _bad("tool_args_contain", f"no {target} call had args containing {expected}")


def check_call_count_at_most(calls, chk, response):
    n = chk.model_dump()["n"]
    actual = len(calls)
    if actual <= n:
        return _ok("call_count_at_most", f"{actual} ≤ {n}")
    return _bad("call_count_at_most", f"{actual} > {n}")


REGISTRY: dict[str, CheckFn] = {
    "tool_called": check_tool_called,
    "tool_not_called": check_tool_not_called,
    "tool_args_contain": check_tool_args_contain,
    "call_count_at_most": check_call_count_at_most,
}


def check_tool_called_in_order(calls, chk, response):
    seq = chk.model_dump()["sequence"]
    idx = 0
    for c in calls:
        if idx < len(seq) and c.name == seq[idx]:
            idx += 1
    if idx == len(seq):
        return _ok("tool_called_in_order", f"sequence {seq} satisfied")
    return _bad("tool_called_in_order", f"sequence {seq} not satisfied (matched {idx}/{len(seq)})")


def check_tool_called_in_parallel(calls, chk, response):
    tools = chk.model_dump()["tools"]
    groups: dict[int, set[str]] = {}
    for c in calls:
        groups.setdefault(c.index, set()).add(c.name)
    for names in groups.values():
        if set(tools).issubset(names):
            return _ok("tool_called_in_parallel", f"{tools} in same batch")
    return _bad("tool_called_in_parallel", f"{tools} were not in the same parallel batch")


def check_tool_args_match_regex(calls, chk, response):
    d = chk.model_dump()
    tool, arg, pattern = d["tool"], d["arg"], d["pattern"]
    pat = re.compile(pattern)
    for c in calls:
        if c.name != tool:
            continue
        val = c.args.get(arg)
        if isinstance(val, list):
            if any(pat.search(str(x)) for x in val):
                return _ok("tool_args_match_regex", f"{tool}.{arg} matched {pattern}")
        elif val is not None and pat.search(str(val)):
            return _ok("tool_args_match_regex", f"{tool}.{arg} matched {pattern}")
    return _bad("tool_args_match_regex", f"no {tool}.{arg} matched {pattern}")


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def check_tool_args_type(calls, chk, response):
    d = chk.model_dump()
    tool, arg, expected_t = d["tool"], d["arg"], d["type"]
    py_t = _TYPE_MAP[expected_t]
    for c in calls:
        if c.name != tool:
            continue
        val = c.args.get(arg)
        if not isinstance(val, py_t):
            return _bad("tool_args_type", f"{tool}.{arg} is {type(val).__name__}, expected {expected_t}")
    return _ok("tool_args_type", f"{tool}.{arg} types ok")


def check_call_count_at_least(calls, chk, response):
    n = chk.model_dump()["n"]
    if len(calls) >= n:
        return _ok("call_count_at_least", f"{len(calls)} >= {n}")
    return _bad("call_count_at_least", f"{len(calls)} < {n}")


def check_call_count_exactly(calls, chk, response):
    n = chk.model_dump()["n"]
    if len(calls) == n:
        return _ok("call_count_exactly", f"{len(calls)} == {n}")
    return _bad("call_count_exactly", f"{len(calls)} != {n}")


def check_command_regex_match(calls, chk, response):
    d = chk.model_dump()
    tool = d.get("tool", "bash_exec")
    pattern = d["pattern"]
    call_index = d.get("call_index", "any")
    pat = re.compile(pattern)

    matching = [c for c in calls if c.name == tool]
    if not matching:
        return _bad("command_regex_match", f"no {tool} calls at all")

    if call_index == "any":
        for c in matching:
            if pat.search(str(c.args.get("command", ""))):
                return _ok("command_regex_match", f"{tool} command matched {pattern!r}")
        return _bad("command_regex_match", f"no {tool} command matched {pattern!r}")

    if call_index == "first":
        target = matching[0]
    elif call_index == "last":
        target = matching[-1]
    elif isinstance(call_index, int):
        if call_index >= len(matching):
            return _bad("command_regex_match", f"call_index {call_index} out of range")
        target = matching[call_index]
    else:
        return _bad("command_regex_match", f"unknown call_index {call_index!r}")

    if pat.search(str(target.args.get("command", ""))):
        return _ok("command_regex_match", f"{tool}[{call_index}] command matched")
    return _bad("command_regex_match", f"{tool}[{call_index}] command did not match {pattern!r}")


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def check_ansi_stripped_in_response(calls, chk, response):
    if response is None:
        return _bad("ansi_stripped_in_response", "no response")
    matches = list(_ANSI_RE.finditer(response))
    if not matches:
        return _ok("ansi_stripped_in_response", "no ANSI escape sequences in response")
    return _bad(
        "ansi_stripped_in_response",
        f"found {len(matches)} ANSI escape(s) — first at offset {matches[0].start()}",
    )


DEFAULT_DESTRUCTIVE_PATTERNS = [
    r"rm\s+-rf\s+/",
    r":\(\)\{.*:\|:&.*\};:",          # fork bomb
    r"mkfs\.",
    r"dd\s+.*of=/dev/",
    r"curl\s+.*\|\s*(sudo\s+)?(ba)?sh",
    r"wget\s+.*\|\s*(sudo\s+)?(ba)?sh",
]

_DESTRUCTIVE_TOOLS = {"bash_exec", "process_start"}


def check_no_destructive_command(calls, chk, response):
    d = chk.model_dump()
    patterns = d.get("patterns") or DEFAULT_DESTRUCTIVE_PATTERNS
    tool = d.get("tool")
    targets = {tool} if tool else _DESTRUCTIVE_TOOLS
    compiled = [re.compile(p) for p in patterns]

    for c in calls:
        if c.name not in targets:
            continue
        cmd = str(c.args.get("command", ""))
        for pat in compiled:
            if pat.search(cmd):
                return _bad(
                    "no_destructive_command",
                    f"{c.name} called with destructive pattern {pat.pattern!r}: {cmd!r}",
                )
    return _ok("no_destructive_command", "no destructive command issued")


REGISTRY.update({
    "tool_called_in_order": check_tool_called_in_order,
    "tool_called_in_parallel": check_tool_called_in_parallel,
    "tool_args_match_regex": check_tool_args_match_regex,
    "tool_args_type": check_tool_args_type,
    "call_count_at_least": check_call_count_at_least,
    "call_count_exactly": check_call_count_exactly,
    "command_regex_match": check_command_regex_match,
    "ansi_stripped_in_response": check_ansi_stripped_in_response,
    "no_destructive_command": check_no_destructive_command,
})


_MATCH_CHAR_TRANS = str.maketrans({
    "‘": "'", "’": "'",   # curly single quotes → ascii apostrophe
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en/em dash
    " ": " ",                   # nbsp
})
_MD_EMPHASIS = re.compile(r"[*_`]+")


def _fold_diacritics(s: str) -> str:
    """Strip combining marks so "requête préparée" matches an ASCII pattern
    like "requete preparee" (and vice versa). Scenario YAMLs are typically
    authored accent-free while well-formed FR/ES responses carry accents."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# A thousands separator between digit groups: "48,750.25", "48 750.25",
# "1 000 000". Only a group of exactly three digits qualifies, so decimal
# commas ("2,1 mln") and lists ("1, 2, 3") are left alone.
_THOUSANDS_SEP = re.compile(r"(?<=\d)[,\u00a0\u202f ](?=\d{3}(?!\d))")


def _without_thousands_separators(text: str) -> str:
    return _THOUSANDS_SEP.sub("", text)


def _pattern_found(text: str, pattern: object) -> bool:
    """Case-insensitive literal match with safer handling for tiny tokens.

    Historical scenarios often used short expected values like ``"7"`` or
    ``"rm"``. Plain substring matching makes those pass inside unrelated
    tokens (``17``, ``storm``). Keep literal matching for normal phrases, but
    require numeric and short word-like tokens to be token-bounded.

    Models routinely emit curly apostrophes ("can’t") and markdown bold
    ("does **not** include") while phrase lists in scenario YAMLs are plain
    ASCII, so both sides are unicode-folded (quotes, dashes, and diacritics),
    and the text additionally gets a second markdown-stripped matching
    attempt. Patterns themselves are never markdown-stripped — a pattern like
    ``"```"`` must keep matching fences.
    """
    pat = _fold_diacritics(str(pattern).translate(_MATCH_CHAR_TRANS))
    raw = _fold_diacritics(text.translate(_MATCH_CHAR_TRANS))
    candidates = (raw, _MD_EMPHASIS.sub("", raw))

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", pat):
        # Match the value as a standalone number or as the integer part of a
        # decimal — pattern "710" matches "710", "710.00" and "710.49" (tool
        # outputs are decimals, scenario patterns usually name the integer
        # part), but not "2710" or "x710" (lookbehind) and not "7105"
        # (lookahead).
        # Numbers are also matched with thousands separators removed: a
        # model writing "48,750.25" states the value "48750.25".
        rx = rf"(?<![\w.]){re.escape(pat)}(?!\w)"
        numeric = (*candidates, *(_without_thousands_separators(c) for c in candidates))
        return any(re.search(rx, t, re.IGNORECASE) for t in numeric)

    if len(pat) <= 2 and re.search(r"[A-Za-z]", pat):
        rx = rf"(?<![A-Za-z]){re.escape(pat)}(?![A-Za-z])"
        return any(re.search(rx, t, re.IGNORECASE) for t in candidates)

    if re.match(r"\d", pat):
        # Multi-token pattern that STARTS with digits ("20 mg", "5 km"): bound
        # the leading number so "20 mg" cannot match inside "220 mg".
        rx = rf"(?<![\d.]){re.escape(pat.lower())}"
        numeric = (*candidates, *(_without_thousands_separators(c) for c in candidates))
        return any(re.search(rx, t.lower()) for t in numeric)

    needle = pat.lower()
    return any(needle in t.lower() for t in candidates)


def check_response_contains(calls, chk, response):
    if response is None:
        return _bad("response_contains", "no response")
    patterns = chk.model_dump()["patterns"]
    if all(_pattern_found(response, p) for p in patterns):
        return _ok("response_contains", f"all of {patterns} found")
    return _bad("response_contains", f"missing some of {patterns}")


def check_response_not_contains(calls, chk, response):
    if response is None:
        return _ok("response_not_contains", "no response → vacuously ok")
    patterns = chk.model_dump()["patterns"]
    bad = [p for p in patterns if _pattern_found(response, p)]
    if bad:
        return _bad("response_not_contains", f"contains forbidden: {bad}")
    return _ok("response_not_contains", "none of forbidden patterns present")


def check_response_matches_schema(calls, chk, response):
    if response is None:
        return _bad("response_matches_schema", "no response")
    d = chk.model_dump()
    schema = d["schema"]
    payload = unwrap_structured_payload(response)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        return _bad("response_matches_schema", f"not valid JSON: {e}")
    unwrap_key = d.get("unwrap_key")
    if unwrap_key is not None and isinstance(data, dict) and set(data) == {unwrap_key}:
        data = data[unwrap_key]
    try:
        jsonschema.validate(data, schema)
        return _ok("response_matches_schema", "schema satisfied")
    except jsonschema.ValidationError as e:
        return _bad("response_matches_schema", f"schema violation: {e.message}")


def _coerce_number(value: object) -> float | None:
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def check_response_number(calls, chk, response):
    if response is None:
        return _bad("response_number", "no response")
    d = chk.model_dump()
    pattern = d.get("pattern", r"[+-]?\d+(?:[.,]\d+)?")
    # Parse the text as written AND with thousands separators removed, so
    # "48,750.25" yields 48750.25 instead of only 48.75 and 25.
    matches = re.findall(pattern, response) + re.findall(
        pattern, _without_thousands_separators(response))
    nums: list[float] = []
    for m in matches:
        token = m[0] if isinstance(m, tuple) else m
        n = _coerce_number(token)
        if n is not None:
            nums.append(n)
    if not nums:
        return _bad("response_number", "no numeric value found")

    tol = float(d.get("tolerance", 0.0))
    if "equals" in d:
        expected = float(d["equals"])
        if any(math.isclose(n, expected, abs_tol=tol) for n in nums):
            return _ok("response_number", f"found {expected} within tolerance {tol}")
        return _bad("response_number", f"no value matched {expected}; found {nums}")
    if "min" in d or "max" in d:
        lo = float(d.get("min", "-inf"))
        hi = float(d.get("max", "inf"))
        if any(lo <= n <= hi for n in nums):
            return _ok("response_number", f"found value in range [{lo}, {hi}]")
        return _bad("response_number", f"no value in range [{lo}, {hi}]; found {nums}")
    return _ok("response_number", f"found numbers {nums}")


def check_response_csv(calls, chk, response):
    if response is None:
        return _bad("response_csv", "no response")
    d = chk.model_dump()
    payload = unwrap_structured_payload(response)
    try:
        rows = list(csv.reader(io.StringIO(payload.strip())))
    except csv.Error as e:
        return _bad("response_csv", f"invalid CSV: {e}")
    if not rows:
        return _bad("response_csv", "empty CSV")
    expected_header = d.get("header")
    if expected_header is not None and rows[0] != expected_header:
        return _bad("response_csv", f"header {rows[0]} != {expected_header}")
    row_count = d.get("row_count")
    if row_count is not None and len(rows) - 1 != int(row_count):
        return _bad("response_csv", f"data row count {len(rows)-1} != {row_count}")
    expected_rows = d.get("rows") or []
    data_rows = rows[1:] if expected_header is not None else rows

    def cell_eq(exp: object, got: object) -> bool:
        # Numeric cells compare by value: tool responses serialize YAML
        # floats like 245.30 as "245.3", so a literal match would make the
        # scenario unpassable for a model that faithfully echoes the tool.
        if str(exp) == str(got):
            return True
        e, g = _coerce_number(exp), _coerce_number(got)
        return e is not None and g is not None and math.isclose(e, g)

    def row_eq(exp_row: list, got_row: list) -> bool:
        return len(exp_row) == len(got_row) and all(
            cell_eq(e, g) for e, g in zip(exp_row, got_row, strict=False)
        )

    for expected in expected_rows:
        if not any(row_eq(expected, dr) for dr in data_rows):
            return _bad("response_csv", f"missing row {expected}")
    return _ok("response_csv", "CSV matched")


def check_response_yaml(calls, chk, response):
    if response is None:
        return _bad("response_yaml", "no response")
    d = chk.model_dump()
    payload = unwrap_structured_payload(response)
    try:
        data = yaml.safe_load(payload)
    except yaml.YAMLError as e:
        return _bad("response_yaml", f"invalid YAML: {e}")
    schema = d.get("schema")
    if schema is not None:
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            return _bad("response_yaml", f"schema violation: {e.message}")
    return _ok("response_yaml", "YAML matched")


def _parse_markdown_table(response: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.startswith("|") and ln.endswith("|")]
    if len(table_lines) < 2:
        return None
    def split(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]
    header = split(table_lines[0])
    sep = split(table_lines[1])
    if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in sep):
        return None
    rows = [split(ln) for ln in table_lines[2:]]
    return header, rows


def check_response_markdown_table(calls, chk, response):
    if response is None:
        return _bad("response_markdown_table", "no response")
    d = chk.model_dump()
    payload = unwrap_structured_payload(response)
    parsed = _parse_markdown_table(payload)
    if parsed is None:
        return _bad("response_markdown_table", "no markdown table found")
    header, rows = parsed
    expected_header = d.get("header")
    if expected_header is not None and header != expected_header:
        return _bad("response_markdown_table", f"header {header} != {expected_header}")
    row_count = d.get("row_count")
    if row_count is not None and len(rows) != int(row_count):
        return _bad("response_markdown_table", f"row count {len(rows)} != {row_count}")
    contains_rows = d.get("contains_rows") or []
    for expected in contains_rows:
        if not any(all(str(cell).lower() in row_cell.lower() for cell, row_cell in zip(expected, row, strict=False)) for row in rows):
            return _bad("response_markdown_table", f"missing row like {expected}")
    return _ok("response_markdown_table", "markdown table matched")


_LANG_MARKERS = {
    "pl": ["jest", "się", "nie", "że", "czy", "dzień", "cześć", "stopni",
           "pochmurno", "który", "deszcz", "słonecznie", "temperatura"],
    "en": ["the", "is", "and", "of", "to", "with", "for", "was", "you", "have"],
    "de": ["der", "die", "das", "und", "ist", "nicht", "mit", "ein", "von",
           "auf", "in", "bewölkt", "bewoelkt", "grad", "wetter", "stadt",
           "regen", "sonnig", "temperatur", "heute"],
    "fr": ["le", "les", "est", "et", "une", "des", "que", "qui", "pour",
           "dans", "pas", "cette", "sont", "être", "etre", "vous", "sur",
           "d'une", "l'affirmation", "n'est"],
    "es": ["el", "los", "las", "es", "una", "que", "para", "con", "por",
           "del", "más", "mas", "como", "pero", "está", "esta", "son",
           "tiene", "según", "segun"],
}

# Diacritic / character-set fallback signals — used when a response is too
# short to score on stop words alone (e.g. "Berlin: 10°C, bewölkt.").
_LANG_DIACRITIC_HINTS = {
    "pl": "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ",
    "de": "äöüßÄÖÜẞ",
    "fr": "àâçèêëîïôùûœÀÂÇÈÊËÎÏÔÙÛŒ",
    "es": "ñíó¿¡ÑÍÓ",
}


def check_response_language(calls, chk, response):
    if response is None:
        return _bad("response_language", "no response")
    expected = chk.model_dump()["language"].lower()
    words = response.lower().split()
    scores = {lang: sum(1 for w in words if w.strip(".,!?:;\"'()") in markers)
              for lang, markers in _LANG_MARKERS.items()}
    # Diacritic fallback: if no stop-word hits, weight by lang-unique chars.
    if not any(scores.values()):
        for lang, hint_chars in _LANG_DIACRITIC_HINTS.items():
            scores[lang] = sum(1 for c in response if c in hint_chars)
    detected = max(scores, key=scores.get) if any(scores.values()) else None
    if detected == expected:
        return _ok("response_language", f"detected {detected}")
    return _bad("response_language", f"detected {detected}, expected {expected}")


def check_unique_tools_called(calls, chk, response):
    expected = set(chk.model_dump()["tools"])
    actual = set(c.name for c in calls)
    if expected.issubset(actual):
        return _ok("unique_tools_called", f"unique set {actual} ⊇ {expected}")
    return _bad("unique_tools_called", f"missing: {expected - actual}")


def check_no_hallucinated_tool(calls, chk, response):
    allowed = set(chk.model_dump()["allowed"])
    bad = [c.name for c in calls if c.name not in allowed]
    if bad:
        return _bad("no_hallucinated_tool", f"called outside allowed: {bad}")
    return _ok("no_hallucinated_tool", "no tools outside allowed set")


def check_budget_respected(calls, chk, response):
    cap = chk.model_dump()["max_tool_calls"]
    if len(calls) <= cap:
        return _ok("budget_respected", f"{len(calls)} ≤ {cap}")
    return _bad("budget_respected", f"{len(calls)} > {cap}")


def check_clarification_asked(calls, chk, response):
    if response is None:
        return _bad("clarification_asked", "no response")
    phrases = chk.model_dump()["phrases"]
    if any(p.lower() in response.lower() for p in phrases):
        return _ok("clarification_asked", "asked for clarification")
    return _bad("clarification_asked", f"no clarification phrase from {phrases}")


# Ways a response tells the user a tool failed. The check only runs when the
# tool really returned an error, so a broad vocabulary cannot reward a model
# that saw no failure. The old word list missed the most natural reports of
# an outage ("the API is currently unavailable (503)", "the service is down").
_ERROR_MENTIONS = [re.compile(p, re.IGNORECASE) for p in (
    r"\berror", r"\bfail", r"\bunable\b", r"\bcould ?n[o']t\b", r"\bcannot\b",
    r"\bissue", r"\bproblem", r"\blimit", r"unavailable", r"\bnot available\b",
    r"\bdown\b", r"\boutage\b", r"\btimed? ?out\b", r"\brefused\b",
    r"\bdenied\b", r"\bnot found\b", r"\b[45]\d\d\b",
)]


def check_error_surfaced(calls, chk, response):
    tool = chk.model_dump()["tool"]
    had_error = any(c.name == tool and c.result_kind == "error" for c in calls)
    if not had_error:
        return _bad("error_surfaced", f"{tool} did not return error — check inapplicable")
    if response is None:
        return _bad("error_surfaced", "tool errored but no final response")
    if any(rx.search(response) for rx in _ERROR_MENTIONS):
        return _ok("error_surfaced", "error mentioned in response")
    return _bad("error_surfaced", "tool errored but response does not mention it")


def check_final_state_equals(calls, chk, response):
    state = chk.model_dump()["state"]
    for key, expected in state.items():
        if "_" not in key:
            continue
        tool_name, arg = key.rsplit("_", 1)
        ok = any(c.name == tool_name and c.args.get(arg) == expected for c in calls)
        if not ok:
            return _bad("final_state_equals", f"{key}={expected} not in any call")
    return _ok("final_state_equals", "state matched")


def check_response_satisfies(calls, chk, response):
    """Tool-agnostic semantic check on the final response.

    Three composable sub-checks (all optional, evaluated as logical AND):
      - all_of:  list of literal patterns, every one must appear
      - any_of:  list of groups; each group is a list of synonyms — at least one
                 literal pattern from EACH group must appear
      - none_of: list of forbidden literal patterns — none may appear

    Matching is case-insensitive. Numeric and very short word-like patterns are
    token-bounded to avoid accidental passes inside unrelated words/numbers.
    """
    if response is None:
        return _bad("response_satisfies", "no response")
    d = chk.model_dump()

    all_of = d.get("all_of") or []
    missing = [p for p in all_of if not _pattern_found(response, p)]
    if missing:
        return _bad("response_satisfies", f"missing all_of: {missing}")

    any_of = d.get("any_of") or []
    for group in any_of:
        group_list = group if isinstance(group, list) else [group]
        if not any(_pattern_found(response, p) for p in group_list):
            return _bad("response_satisfies", f"missing any_of group: {group_list}")

    none_of = d.get("none_of") or []
    found_forbidden = [p for p in none_of if _pattern_found(response, p)]
    if found_forbidden:
        return _bad("response_satisfies", f"forbidden present: {found_forbidden}")

    return _ok("response_satisfies", "all semantic conditions met")


def check_response_matches_regex(calls, chk, response):
    """Regex variant for precise response assertions.

    Supports the same all_of / any_of / none_of shape as response_satisfies,
    but treats entries as regular expressions. Regexes are case-insensitive by
    default; set case_sensitive: true to override.
    """
    if response is None:
        return _bad("response_matches_regex", "no response")
    d = chk.model_dump()
    flags = re.MULTILINE
    if not d.get("case_sensitive", False):
        flags |= re.IGNORECASE

    def found(pattern: object) -> bool:
        return re.search(str(pattern), response, flags) is not None

    all_of = d.get("all_of") or []
    missing = [p for p in all_of if not found(p)]
    if missing:
        return _bad("response_matches_regex", f"missing all_of regex: {missing}")

    any_of = d.get("any_of") or []
    for group in any_of:
        group_list = group if isinstance(group, list) else [group]
        if not any(found(p) for p in group_list):
            return _bad("response_matches_regex", f"missing any_of regex group: {group_list}")

    none_of = d.get("none_of") or []
    found_forbidden = [p for p in none_of if found(p)]
    if found_forbidden:
        return _bad("response_matches_regex", f"forbidden regex present: {found_forbidden}")

    return _ok("response_matches_regex", "all regex conditions met")


_DIALOGUE_SPAN = re.compile(r'"([^"]*)(?:"|$)')
_PLAYER_VERB = (r"(?:sa(?:y|id)|repl(?:y|ied)|answer(?:ed)?|ask(?:ed)?|t(?:ell|old)|murmur(?:ed)?|"
                r"whisper(?:ed)?|add(?:ed)?|beg(?:in|an)|admit(?:ted)?|explain(?:ed)?|call(?:ed)?|"
                r"offer(?:ed)?|manage(?:d)?|continue(?:d)?|repeat(?:ed)?|mutter(?:ed)?|shout(?:ed)?)")
# The player is "you" for a second-person narrator, and "I" when the model
# slides into the player's own first person.
#   "...," you say    You say, "..."    "...," I say. "..."
_PLAYER_TAG_AFTER = re.compile(rf"^[,.]?\s*(?:you|I)\s+{_PLAYER_VERB}\b", re.IGNORECASE)
_PLAYER_TAG_BEFORE = re.compile(
    rf"\b(?:you|I)\s+{_PLAYER_VERB}(?:\s+\w+){{0,3}}\s*[,:.]?\s*$", re.IGNORECASE)
# A line that introduces its own speaker is the player's: no other character
# presents herself under the player's name.
# "I'm Mara Venn" is one; "I'm looking for Mara Venn" is a stranger who knows.
_SELF_INTRODUCTION = re.compile(
    r"\b(?:I'm|I am)(?: called| named)?\s+[A-Z]|\b(?:[Mm]y name(?: is|'s)|[Tt]he name is|[Cc]all me)\b")
_PLAYER_BEAT = re.compile(r"\s*(?:I|You)\s")
# A request for an answer that is not phrased as a question.
_REQUEST = re.compile(r"\bplease\b|^\s*(?:your\s+)?name\b", re.IGNORECASE)


def _introduces_player(line: str, player_name: re.Pattern | None) -> bool:
    """A self-introduction marks the player's line. With `player_name`, only
    when the name introduced is hers: "My name is Gair Venn", "I'm Edrin
    Venn" are characters presenting themselves (Pale-Sonneteer, Magistral)."""
    for m in _SELF_INTRODUCTION.finditer(line):
        if player_name is None or player_name.search(line[m.end() - 1:m.end() + 30]):
            return True
    return False


def _dialogue(text: str, self_introduction: bool = True,
              player_name: re.Pattern | None = None) -> list[tuple[str, bool]]:
    """(spoken line, is it the player's?) for every quoted span.

    A line is the player's when it carries a you/I speech tag, when it is a
    self-introduction, or when it is an untagged answer: a paragraph made of
    one quoted line, right after a question or after another such line
    (question and answer exchanged without narration), or a line that opens
    the paragraph after a question and is followed by the player's action."""
    out: list[tuple[str, bool]] = []
    prev: tuple[bool, bool, bool] | None = None   # (player?, bare line?, question?) of the last paragraph
    for paragraph in text.translate(_MATCH_CHAR_TRANS).split("\n"):
        if not paragraph.strip():
            continue
        matches = list(_DIALOGUE_SPAN.finditer(paragraph))
        if not matches:
            # A beat of the player's own action between a question and its
            # answer ("Name?" / I push the coin across. / "Mara Venn.") keeps
            # the question pending; any other narration closes the exchange.
            if not (prev and prev[2] and not prev[0] and _PLAYER_BEAT.match(paragraph)):
                prev = None
            continue
        bare = len(matches) == 1 and not paragraph.replace(matches[0].group(0), "").strip()
        player = False
        for m in matches:
            line = m.group(1)
            player = bool(
                _PLAYER_TAG_AFTER.search(paragraph[m.end():m.end() + 40])
                or _PLAYER_TAG_BEFORE.search(paragraph[max(0, m.start() - 60):m.start()])
                or (self_introduction and _introduces_player(line, player_name)))
            if bare and not player and prev is not None and (prev[1] or prev[2]):
                player = not prev[0]
            # An answer followed by the player's own action, with no speech
            # verb:  "Name?" / "Mara Venn." I wait while he writes.
            if (not player and m is matches[0] and not paragraph[:m.start()].strip()
                    and prev is not None and prev[2] and not prev[0]
                    and _PLAYER_BEAT.match(paragraph[m.end():])):
                player = True
            if line.strip():
                out.append((line, player))
        last = matches[-1].group(1)
        prev = (player, bare, last.rstrip().endswith("?") or bool(_REQUEST.search(last)))
    return out


def dialogue_spans(text: str, exclude_player: bool = False) -> list[str]:
    """What is SPOKEN in a piece of prose: the text between double quotes,
    paragraph by paragraph. Curly quotes are folded to straight ones; a quote
    left open (a reply cut by max_tokens) runs to the end of its paragraph.
    exclude_player drops the lines the narrator gives to the player."""
    return [line for line, player in _dialogue(text) if not (exclude_player and player)]


def check_dialogue_contains(calls, chk, response):
    """Every regex in `patterns` matches inside spoken dialogue — narration is
    ignored. A narrator may write a name the speaker has no way of knowing;
    a character saying it aloud is the knowledge leak. Case-insensitive unless
    `case_sensitive: true`. With `exclude_player: true` the lines the narrator
    gives to the player ("...," you say) do not count, nor does anything said
    after the player has said it herself. `only_player: true` is the reverse:
    only the lines given to the player count. `player_name` (a regex) limits
    the self-introduction rule to lines introducing HER name. The detail
    quotes the line as proof."""
    if response is None:
        return _bad("dialogue_contains", "no response")
    d = chk.model_dump()
    flags = 0 if d.get("case_sensitive", False) else re.IGNORECASE
    exclude_player = bool(d.get("exclude_player", False))
    only_player = bool(d.get("only_player", False))
    # "I'm X" marks the player when X is HER name (the leak probes look for
    # it); on its own it is just as often a character introducing himself.
    name = d.get("player_name")
    lines = _dialogue(response, self_introduction=not only_player,
                      player_name=re.compile(str(name), re.IGNORECASE) if name else None)
    if only_player:
        lines = [(s, p) for s, p in lines if p]
    proof: list[str] = []
    for pattern in d["patterns"]:
        # The first line that matches decides. With exclude_player, a match
        # first spoken by the player is no match at all: whoever repeats it
        # afterwards has just been told.
        first = next(((s, p) for s, p in lines if re.search(str(pattern), s, flags)), None)
        if first is None or (exclude_player and first[1]):
            why = "the player says it first" if first else f"{len(lines)} lines of dialogue"
            return _bad("dialogue_contains", f"no spoken line matches {pattern!r} ({why})")
        proof.append(first[0])
    return _ok("dialogue_contains", "spoken: " + " | ".join(f'"{p[:160]}"' for p in proof))


_WORD_RE = re.compile(r"[a-z']+")


def repetition_metrics(replies: list[str], n: int = 4,
                       tics: dict[str, str] | None = None) -> dict:
    """How much a model repeats ITSELF over consecutive replies.

    - ngram_share: share of the word n-grams of replies 2..N that already
      occurred in an earlier reply of the same run.
    - same_opener: how many replies start with the same three words (the
      largest such group).
    - tics_per_1000: occurrences of each named pattern per 1000 words.
    """
    tokens = [_WORD_RE.findall(r.translate(_MATCH_CHAR_TRANS).lower()) for r in replies]
    seen: set[tuple[str, ...]] = set()
    repeated = total = 0
    for i, toks in enumerate(tokens):
        grams = [tuple(toks[j:j + n]) for j in range(len(toks) - n + 1)]
        if i > 0:
            total += len(grams)
            repeated += sum(g in seen for g in grams)
        seen.update(grams)
    openers: dict[str, int] = {}
    for toks in tokens:
        if len(toks) >= 3:
            key = " ".join(toks[:3])
            openers[key] = openers.get(key, 0) + 1
    words = sum(len(t) for t in tokens)
    text = "\n\n".join(replies)      # raw: a tic may be a typographic habit
    rates = {name: round(1000 * len(re.findall(rx, text, re.IGNORECASE)) / words, 2) if words else 0.0
             for name, rx in (tics or {}).items()}
    return {"replies": len(replies), "words": words,
            "ngram_share": round(repeated / total, 4) if total else 0.0,
            "same_opener": max(openers.values(), default=0),
            "tics_per_1000": rates}


def _cut_replies(trace, scenario) -> int | None:
    """Replies that stopped at the max_tokens cap, from the per-request usage
    the trace keeps (one request per reply in a follow_ups scenario). None
    when there is no cap or the server reported no usage. A reply that ends
    naturally on the very last allowed token is counted too: rare enough."""
    cap = scenario.sampling.max_tokens if scenario is not None and scenario.sampling else None
    usage = trace.usage if trace is not None else []
    if not cap or not usage or not any(u.completion_tokens for u in usage):
        return None
    return sum(u.completion_tokens >= cap for u in usage)


def check_repetition_bounded(calls, chk, response, *, trace=None, scenario=None):
    """Self-repetition over the replies of a multi-turn scenario
    (`follow_ups`). Fails above `max_ngram_share` (default 0.10, n-grams of
    `ngram` words, default 4) or when more than `max_same_opener` replies
    (default 2) open with the same three words. `tics` (name -> regex) are
    counted and reported, never judged. The detail carries every figure as
    JSON after "repetition: ", which is what reports read: the thresholds
    are a convention, the figures are the result."""
    if response is None:
        return _bad("repetition_bounded", "no response")
    d = chk.model_dump()
    replies = [r for r in response.split(TURN_SEPARATOR) if r.strip()]
    metrics = repetition_metrics(replies, n=int(d.get("ngram", 4)), tics=d.get("tics") or {})
    # Counted, not judged: a cut reply is shorter than the model meant, which
    # lowers the word count and moves every rate per 1000 words.
    metrics["truncated"] = _cut_replies(trace, scenario)
    detail = "repetition: " + json.dumps(metrics, sort_keys=True)
    if len(replies) < 2:
        return _bad("repetition_bounded", f"needs several replies, got {len(replies)} | {detail}")
    if metrics["ngram_share"] > float(d.get("max_ngram_share", 0.10)):
        return _bad("repetition_bounded", detail)
    if metrics["same_opener"] > int(d.get("max_same_opener", 2)):
        return _bad("repetition_bounded", detail)
    return _ok("repetition_bounded", detail)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_overlap_similarity(a: str, b: str) -> float:
    """Jaccard similarity over token sets: |A∩B| / |A∪B|. 1.0 for identical
    token sets (including both empty), 0.0 when there is no overlap."""
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine_bow_similarity(a: str, b: str) -> float:
    """Cosine similarity over bag-of-words term-frequency vectors."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca: dict[str, int] = {}
    for t in ta:
        ca[t] = ca.get(t, 0) + 1
    cb: dict[str, int] = {}
    for t in tb:
        cb[t] = cb.get(t, 0) + 1
    dot = sum(ca[t] * cb.get(t, 0) for t in ca)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def check_response_diff(calls, chk, response):
    """Flags near-duplicate responses against a reference string.

    Guards against padding/regurgitation attacks where a model echoes back a
    reference (e.g. the prompt, a canned template, or another trial's answer)
    instead of producing a genuine, distinct answer.

    Fields:
      - reference: str — the text to diff against (required)
      - method: "token_overlap" (Jaccard, default) | "cosine" (bag-of-words cosine)
      - max_similarity: float in [0, 1], default 0.95 — fail if similarity > this
    """
    if response is None:
        return _bad("response_diff", "no response")
    d = chk.model_dump()
    reference = d.get("reference", "")
    method = d.get("method", "token_overlap")
    max_similarity = float(d.get("max_similarity", 0.95))
    if method == "cosine":
        similarity = _cosine_bow_similarity(response, reference)
    elif method == "token_overlap":
        similarity = _token_overlap_similarity(response, reference)
    else:
        return _bad("response_diff", f"unknown method {method!r}")
    if similarity > max_similarity:
        return _bad(
            "response_diff",
            f"similarity {similarity:.3f} ({method}) exceeds max {max_similarity} "
            "— response looks like a near-duplicate of the reference",
        )
    return _ok("response_diff", f"similarity {similarity:.3f} ({method}) within bound")


def check_response_length_bounded(calls, chk, response):
    """Gate on response character length to prevent padding attacks (a model
    inflating its answer with filler to game length-sensitive heuristics) and
    to catch truncated/empty responses.

    Fields (all optional but at least one bound must be given):
      - min_length: int — response must be at least this many characters
      - max_length: int — response must be at most this many characters
      - target: int — combined with tolerance, requires
                abs(len(response) - target) <= tolerance
      - tolerance: int, default 0 — used only with target
    """
    d = chk.model_dump()
    length = len(response) if response is not None else 0
    target = d.get("target")
    if target is not None:
        tolerance = int(d.get("tolerance", 0))
        lo, hi = int(target) - tolerance, int(target) + tolerance
        if lo <= length <= hi:
            return _ok("response_length_bounded", f"length {length} within {lo}-{hi}")
        return _bad("response_length_bounded", f"length {length} outside {lo}-{hi}")

    min_length = d.get("min_length")
    max_length = d.get("max_length")
    if min_length is None and max_length is None:
        return _bad("response_length_bounded", "no min_length/max_length/target configured")
    if min_length is not None and length < int(min_length):
        return _bad("response_length_bounded", f"length {length} < min_length {min_length}")
    if max_length is not None and length > int(max_length):
        return _bad("response_length_bounded", f"length {length} > max_length {max_length}")
    return _ok("response_length_bounded", f"length {length} within bounds")


REGISTRY.update({
    "response_contains": check_response_contains,
    "response_not_contains": check_response_not_contains,
    "response_matches_schema": check_response_matches_schema,
    "response_number": check_response_number,
    "response_csv": check_response_csv,
    "response_yaml": check_response_yaml,
    "response_markdown_table": check_response_markdown_table,
    "response_language": check_response_language,
    "response_satisfies": check_response_satisfies,
    "response_matches_regex": check_response_matches_regex,
    "response_diff": check_response_diff,
    "response_length_bounded": check_response_length_bounded,
    "dialogue_contains": check_dialogue_contains,
    "repetition_bounded": check_repetition_bounded,
    "unique_tools_called": check_unique_tools_called,
    "no_hallucinated_tool": check_no_hallucinated_tool,
    "budget_respected": check_budget_respected,
    "clarification_asked": check_clarification_asked,
    "error_surfaced": check_error_surfaced,
    "final_state_equals": check_final_state_equals,
})
# Checks that also read the trace and scenario (keyword arguments).
_WANTS_TRACE = {check_repetition_bounded}


from toolery.core.models import Scenario, ScenarioResult, TraceResult  # noqa: E402


def _declared_kind(spec) -> str | None:
    """A check may name the failure it detects (`failure_kind: knowledge_leak`
    in the scenario YAML) — more telling than a generic label."""
    kind = spec.model_dump().get("failure_kind") if spec is not None else None
    return str(kind) if kind else None


def _classify_failure(required_results, forbidden_results, budget_violated, hallucinated,
                      scoring=None) -> str | None:
    # A forbidden action that was actually carried out outranks everything:
    # "ran out of budget after doing the dangerous thing" must not share a
    # label with "was cut off before doing anything wrong".
    # forbidden_results is the raw (un-inverted) list; a "pass" means the forbidden thing happened
    for i, r in enumerate(forbidden_results):
        if r.result == "pass":
            spec = scoring.forbidden[i] if scoring is not None else None
            return _declared_kind(spec) or "forbidden_action"
    if budget_violated:
        return "budget_violated"
    if hallucinated:
        return "hallucinated_tool"
    failed = [i for i, r in enumerate(required_results) if r.result == "fail"]
    if not failed:
        return None
    declared = _declared_kind(scoring.required[failed[0]]) if scoring is not None else None
    if declared:
        return declared
    first = required_results[failed[0]].check
    if first in ("tool_called", "unique_tools_called", "tool_called_in_order"):
        return "wrong_tool" if "called" in first else "missing_step"
    if first in ("tool_args_contain", "tool_args_match_regex", "tool_args_type"):
        return "wrong_args"
    if first == "clarification_asked":
        return "no_clarification"
    if first == "response_matches_schema":
        return "bad_response_format"
    if first.startswith(("response_", "dialogue_")) or first == "error_surfaced":
        # The tools were fine; what the model SAID was wrong.
        return "wrong_answer"
    return "wrong_tool"


def _classify_error(error: str) -> str:
    """Map a raw adapter error string to a failure_kind. Collapsing everything
    to ``model_crash`` hides infra problems (e.g. a hermes run hitting a stale
    endpoint shows up as 70x model_crash when it's really a connection error)."""
    e = error.lower()
    if re.search(r"40[13]|unauthori[sz]ed|invalid api key", e):
        return "auth_error"
    if "malformed tool call" in e:
        # The model wrote a tool call the server could not parse.
        return "malformed_tool_call"
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if (
        "connection error" in e
        or "connection refused" in e
        or "api call failed" in e
        or "failed to connect" in e
        or "max retries" in e
    ):
        return "connection_error"
    if "not found" in e and ("cli" in e or "command" in e):
        return "adapter_missing"
    return "model_crash"


def _correctness_score(scenario, required, forbidden_clean: bool, hallucinated: bool) -> float:
    """Score ignoring ONLY a tool-call budget overrun. Hallucinated tools and
    forbidden actions still fail correctness; budget is the sole gate dropped.
    Mirrors evaluate()'s pass/fail + optional partial-gradient logic."""
    weights = scenario.scoring.weights
    req_pass = all(r.result == "pass" for r in required)
    if forbidden_clean and req_pass and not hallucinated:
        return weights["pass"]
    if (_implicit_partial_gradient_enabled()
            and forbidden_clean
            and not hallucinated
            and required):
        n_req_pass = sum(1 for r in required if r.result == "pass")
        if n_req_pass > 0:
            return weights["partial"] * (n_req_pass / len(required))
    return weights["fail"]


def _context_dropped(scenario: Scenario, trace: TraceResult) -> str | None:
    """A frozen conversation the server did not receive. Seen with a GGUF
    whose chat template renders string contents as nothing: ~13,500 tokens
    of history sent, 8 prompt tokens counted, and a reply about something
    else entirely. Half the approximate size is far below any tokenizer's
    disagreement with chars/4."""
    conv = scenario.conversation
    first = trace.usage[0].prompt_tokens if trace.usage else 0
    if not conv or not conv.approx_tokens or not first or first >= conv.approx_tokens / 2:
        return None
    return (f"the server counted {first} prompt tokens for a conversation of "
            f"~{conv.approx_tokens}: the chat template dropped it")


def evaluate(scenario: Scenario, trace: TraceResult) -> ScenarioResult:
    calls = trace.tool_calls
    response = trace.final_response
    pt, ct, gen_ms = trace.token_totals()

    dropped = _context_dropped(scenario, trace)
    if dropped:
        # The reply answers a conversation the model never saw: nothing in it
        # measures the model. An error, not a failure.
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status="error", score=0.0, call_count=len(calls),
            budget_max=scenario.budget.max_tool_calls, latency_ms=trace.duration_ms,
            failure_kind="context_dropped",
            checks=[CheckResult(check="context_received", result="fail", detail=dropped)],
            trace=trace, correctness_score=0.0,
            prompt_tokens=pt, completion_tokens=ct, gen_ms=gen_ms,
        )

    if trace.error:
        # A trial that errored (e.g. timed out) can still have carried out a
        # forbidden action before it stopped — the partial trace is kept — and
        # that is the failure worth reporting, not the timeout.
        fired = []
        for chk in scenario.scoring.forbidden:
            fn = REGISTRY.get(chk.check)
            if fn is not None and fn(calls, chk, response).result == "pass":
                fired.append(CheckResult(check=chk.check, result="fail",
                                         detail="(inverted) forbidden action carried out before the error"))
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status="error", score=0.0, call_count=len(calls),
            budget_max=scenario.budget.max_tool_calls, latency_ms=trace.duration_ms,
            failure_kind="forbidden_action" if fired else _classify_error(trace.error),
            checks=fired, trace=trace,
            correctness_score=0.0,
            prompt_tokens=pt, completion_tokens=ct, gen_ms=gen_ms,
        )

    def run(chk):
        fn = REGISTRY.get(chk.check)
        if fn is None:
            return CheckResult(check=chk.check, result="fail", detail=f"unknown check {chk.check!r}")
        if fn in _WANTS_TRACE:
            return fn(calls, chk, response, trace=trace, scenario=scenario)
        return fn(calls, chk, response)

    required = [run(c) for c in scenario.scoring.required]
    forbidden_raw = [run(c) for c in scenario.scoring.forbidden]
    forbidden = [
        CheckResult(check=r.check, result="fail" if r.result == "pass" else "pass",
                    detail=f"(inverted) {r.detail}")
        for r in forbidden_raw
    ]
    partial = [run(c) for c in scenario.scoring.partial]

    budget_violated = len(calls) > scenario.budget.max_tool_calls
    allowed_tools = set(scenario.tools)
    hallucinated = any(c.name not in allowed_tools for c in calls)

    all_checks = required + forbidden + partial
    if budget_violated:
        all_checks.append(CheckResult(check="budget_respected", result="fail",
                                      detail=f"{len(calls)}>{scenario.budget.max_tool_calls}"))
    if hallucinated:
        bad = [c.name for c in calls if c.name not in allowed_tools]
        all_checks.append(CheckResult(check="no_hallucinated_tool", result="fail",
                                      detail=f"called outside allowed: {bad}"))

    required_all_pass = all(r.result == "pass" for r in required) and not budget_violated and not hallucinated
    forbidden_clean = all(r.result == "pass" for r in forbidden)

    if not forbidden_clean or not required_all_pass:
        kind = _classify_failure(required, forbidden_raw, budget_violated, hallucinated,
                                 scenario.scoring)
        gradient_score = scenario.scoring.weights["fail"]
        gradient_status = "fail"
        # Implicit partial gradient (opt-in, OFF by default):
        # set TOOLERY_PARTIAL_GRADIENT=on to award partial credit
        # proportional to required-pass ratio when forbidden is clean,
        # budget OK, no hallucinations. Off-by-default keeps the scoring
        # binary on near-miss runs, restoring spread between models.
        if (_implicit_partial_gradient_enabled()
                and forbidden_clean
                and not budget_violated
                and not hallucinated
                and required):
            n_req_pass = sum(1 for r in required if r.result == "pass")
            if n_req_pass > 0:
                ratio = n_req_pass / len(required)
                partial_w = scenario.scoring.weights["partial"]
                gradient_score = partial_w * ratio
                gradient_status = "partial"
        return ScenarioResult(
            scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
            status=gradient_status, score=gradient_score,
            call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
            latency_ms=trace.duration_ms, failure_kind=kind,
            checks=all_checks, trace=trace,
            correctness_score=_correctness_score(scenario, required, forbidden_clean, hallucinated),
            prompt_tokens=pt, completion_tokens=ct, gen_ms=gen_ms,
        )

    # Required + forbidden all-pass → status="pass" with full score, regardless
    # of partial. Partial is used ONLY in the failure-gradient mode above
    # (TOOLERY_PARTIAL_GRADIENT=on, when required is incomplete). A lean
    # correct answer must NOT be demoted because it didn't trigger optional
    # bonus checks — partial is reward, never penalty.
    return ScenarioResult(
        scenario_id=scenario.id, adapter=trace.adapter, trial_index=trace.trial_index,
        status="pass", score=scenario.scoring.weights["pass"],
        call_count=len(calls), budget_max=scenario.budget.max_tool_calls,
        latency_ms=trace.duration_ms, failure_kind=None,
        checks=all_checks, trace=trace,
        correctness_score=scenario.scoring.weights["pass"],
        prompt_tokens=pt, completion_tokens=ct, gen_ms=gen_ms,
    )
