import math

import numpy as np
import pytest

from qcv.utils import (
    alphabet_fingerprint,
    canonical_decimal,
    first_argmin,
    logsumexp,
    normalize_text,
    read_jsonl,
    softmax,
    stable_hash,
    write_jsonl,
)


def test_normalize_text_collapses_whitespace_case_and_trailing_punctuation():
    assert normalize_text("  How   many RED cubes are there ?! ") == "how many red cubes are there"
    assert normalize_text("a \N{EN DASH} b") == "a - b"


def test_stable_hash_is_deterministic_and_separator_safe():
    assert stable_hash(["ab", "c"]) == stable_hash(["ab", "c"])
    assert stable_hash(["ab", "c"]) != stable_hash(["a", "bc"])
    assert alphabet_fingerprint(("yes", "no")) != alphabet_fingerprint(("no", "yes"))


@pytest.mark.parametrize(
    ("value", "precision", "expected"),
    [(3.0, 0, "3"), ("2.50", None, "2.5"), (-0.0, 0, "0"), (0.125, 2, "0.13"), (1e-3, 3, "0.001")],
)
def test_canonical_decimal(value, precision, expected):
    assert canonical_decimal(value, precision) == expected


def test_canonical_decimal_rejects_non_finite():
    with pytest.raises(ValueError):
        canonical_decimal("nan")


def test_softmax_hand_computed_and_temperature():
    probabilities = softmax(np.array([0.0, math.log(3.0)]))
    assert np.allclose(probabilities, [0.25, 0.75])
    # T = 2 halves the log-odds: odds sqrt(3) : 1.
    tempered = softmax(np.array([0.0, math.log(3.0)]), temperature=2.0)
    assert np.isclose(tempered[1] / tempered[0], math.sqrt(3.0))
    assert np.isclose(softmax(np.array([1000.0, 1000.0]))[0], 0.5)
    with pytest.raises(ValueError):
        softmax(np.array([0.0, 1.0]), temperature=0.0)


def test_logsumexp_matches_direct_formula():
    values = np.array([[0.0, 1.0, 2.0], [-5.0, 0.0, 5.0]])
    assert np.allclose(logsumexp(values, axis=1), np.log(np.sum(np.exp(values), axis=1)))


def test_first_argmin_prefers_first_of_ties():
    assert first_argmin([2.0, 1.0, 1.0, 3.0]) == 1
    assert first_argmin([1.0, 1.0 - 1e-13], atol=1e-12) == 0
    with pytest.raises(ValueError):
        first_argmin([])


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "é"}]
    write_jsonl(path, rows)
    assert list(read_jsonl(path)) == rows
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(read_jsonl(path))


def test_gzip_jsonl_round_trip_is_reproducible(tmp_path):
    rows = [{"instance_id": f"id-{index}", "value": index} for index in range(50)]
    first, second = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    write_jsonl(first, rows)
    write_jsonl(second, rows)
    assert list(read_jsonl(first)) == rows
    assert first.read_bytes() == second.read_bytes()  # no timestamp or file name in the header
    assert first.read_bytes()[:2] == b"\x1f\x8b"
    plain = tmp_path / "a.jsonl"
    write_jsonl(plain, rows)
    assert list(read_jsonl(plain)) == rows
