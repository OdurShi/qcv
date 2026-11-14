"""Small deterministic helpers: text normalisation, hashing, JSON(L) I/O, numerics."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np

_WHITESPACE = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s\?\!\.]+$")


def normalize_text(text: str) -> str:
    """Normalize question text for deterministic matching and deduplication."""
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = _WHITESPACE.sub(" ", text.strip().lower())
    return _TRAILING_PUNCT.sub("", text)


def stable_hash(parts: Iterable[str], length: int = 16) -> str:
    """Process-independent hex digest of a sequence of strings."""
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def alphabet_fingerprint(alphabet: Sequence[str]) -> str:
    return stable_hash(alphabet, length=20)


def canonical_decimal(value: Any, precision: int | None = None) -> str:
    """Return a canonical finite decimal string without scientific notation."""
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Not a finite decimal value: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"Not a finite decimal value: {value!r}")
    if precision is not None:
        quantum = Decimal(1).scaleb(-precision)
        dec = dec.quantize(quantum, rounding=ROUND_HALF_UP)
    normalized = format(dec, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized in {"-0", ""}:
        normalized = "0"
    return normalized


def decimal_value(value: str) -> Decimal:
    try:
        out = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Answer {value!r} is not numeric") from exc
    if not out.is_finite():
        raise ValueError(f"Answer {value!r} is not finite")
    return out


def json_dump(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def json_load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_gzip(path: Path) -> bool:
    return path.suffix == ".gz"


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of a JSON-lines file (``.jsonl`` or ``.jsonl.gz``)."""
    path = Path(path)
    opener = gzip.open if _is_gzip(path) else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write JSON lines atomically; a ``.gz`` suffix selects reproducible gzip output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    if _is_gzip(path):
        # Fixed header fields (no file name, zero timestamp) make the bytes reproducible.
        with (
            tmp.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as z,
        ):
            z.write(payload)
    else:
        tmp.write_bytes(payload)
    tmp.replace(path)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable ``softmax(values / temperature)`` of a one-dimensional array."""
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError(f"Temperature must be positive and finite, got {temperature}")
    values = np.asarray(values, dtype=np.float64) / temperature
    values = values - np.max(values)
    exp = np.exp(values)
    denom = float(np.sum(exp))
    if denom <= 0 or not math.isfinite(denom):
        raise FloatingPointError("Softmax normalization failed")
    return exp / denom


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(shifted, axis=axis))


def first_argmin(values: Sequence[float], atol: float = 0.0) -> int:
    """Index of the minimum; among values within ``atol`` the first one wins."""
    if not values:
        raise ValueError("Cannot select from an empty sequence")
    best_index = 0
    best_value = float(values[0])
    for index, value in enumerate(values[1:], start=1):
        scalar = float(value)
        if scalar < best_value - atol:
            best_index = index
            best_value = scalar
    return best_index
