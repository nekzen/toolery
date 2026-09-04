"""Scorer checks: response_diff (near-duplicate detection) and
response_length_bounded (padding-attack gate)."""
from toolery.core.models import ScoringCheck
from toolery.core.scorer import check_response_diff, check_response_length_bounded


def _chk(**kwargs):
    return ScoringCheck.model_validate({"check": "response_diff", **kwargs})


def _len_chk(**kwargs):
    return ScoringCheck.model_validate({"check": "response_length_bounded", **kwargs})


# ── response_diff ───────────────────────────────────────────────────────────

def test_response_diff_identical_text_fails_default_bound():
    ref = "The capital of France is Paris."
    chk = _chk(reference=ref)
    r = check_response_diff([], chk, ref)
    assert r.result == "fail"


def test_response_diff_distinct_text_passes():
    ref = "The capital of France is Paris."
    response = "Bananas are a good source of potassium and fiber."
    chk = _chk(reference=ref)
    r = check_response_diff([], chk, response)
    assert r.result == "pass"


def test_response_diff_cosine_method():
    ref = "quick brown fox jumps over the lazy dog"
    response = "quick brown fox jumps over the lazy dog"
    chk = _chk(reference=ref, method="cosine", max_similarity=0.9)
    r = check_response_diff([], chk, response)
    assert r.result == "fail"


def test_response_diff_respects_custom_max_similarity():
    ref = "alpha beta gamma delta"
    response = "alpha beta gamma epsilon"  # 3/5 tokens shared -> jaccard 0.6
    # Bound tight enough to fail...
    assert check_response_diff([], _chk(reference=ref, max_similarity=0.5), response).result == "fail"
    # ...and loose enough to pass.
    assert check_response_diff([], _chk(reference=ref, max_similarity=0.9), response).result == "pass"


def test_response_diff_no_response_fails():
    chk = _chk(reference="x")
    assert check_response_diff([], chk, None).result == "fail"


def test_response_diff_unknown_method_fails():
    chk = _chk(reference="x", method="bogus")
    assert check_response_diff([], chk, "x").result == "fail"


# ── response_length_bounded ─────────────────────────────────────────────────

def test_response_length_bounded_min_length_pass():
    chk = _len_chk(min_length=5)
    assert check_response_length_bounded([], chk, "hello world").result == "pass"


def test_response_length_bounded_min_length_fail():
    chk = _len_chk(min_length=20)
    assert check_response_length_bounded([], chk, "short").result == "fail"


def test_response_length_bounded_max_length_fail_padding_attack():
    chk = _len_chk(max_length=10)
    padded = "x" * 500
    assert check_response_length_bounded([], chk, padded).result == "fail"


def test_response_length_bounded_max_length_pass():
    chk = _len_chk(max_length=100)
    assert check_response_length_bounded([], chk, "concise answer").result == "pass"


def test_response_length_bounded_target_tolerance_pass():
    chk = _len_chk(target=10, tolerance=2)
    assert check_response_length_bounded([], chk, "1234567890").result == "pass"  # exactly 10
    assert check_response_length_bounded([], chk, "123456789").result == "pass"   # 9, within tol
    assert check_response_length_bounded([], chk, "12345678901").result == "pass"  # 11, within tol


def test_response_length_bounded_target_tolerance_fail():
    chk = _len_chk(target=10, tolerance=1)
    assert check_response_length_bounded([], chk, "1234").result == "fail"


def test_response_length_bounded_no_bounds_configured_fails():
    chk = _len_chk()
    assert check_response_length_bounded([], chk, "anything").result == "fail"


def test_response_length_bounded_none_response_counts_as_zero_length():
    chk = _len_chk(min_length=1)
    assert check_response_length_bounded([], chk, None).result == "fail"
    chk2 = _len_chk(max_length=0)
    assert check_response_length_bounded([], chk2, None).result == "pass"
