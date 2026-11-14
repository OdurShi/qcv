"""Split a JSONL file of records into a calibration part and a test part.

The assignment of a record depends only on ``--seed`` and its ``instance_id``
(a hash), so it is reproducible and independent of the order of the input.

    python scripts/split_records.py --input data/clevr/clevr_val_raw.jsonl \\
        --calibration-fraction 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qcv.utils import read_jsonl, stable_hash, write_jsonl

HASH_BUCKETS = 16**8


def in_calibration(instance_id: str, fraction: float, seed: int) -> bool:
    """Whether a record belongs to the calibration part."""
    bucket = int(stable_hash([str(seed), "split", instance_id], length=8), 16)
    return bucket < fraction * HASH_BUCKETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--calibration-output", help="default: <input stem>.calibration.jsonl")
    parser.add_argument("--test-output", help="default: <input stem>.test.jsonl")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error("--calibration-fraction must lie strictly between 0 and 1")

    source = Path(args.input)
    calibration_path = Path(
        args.calibration_output or source.with_name(f"{source.stem}.calibration.jsonl")
    )
    test_path = Path(args.test_output or source.with_name(f"{source.stem}.test.jsonl"))
    calibration, test = [], []
    for row in read_jsonl(source):
        instance_id = str(row.get("instance_id", row.get("id", "")))
        if not instance_id:
            raise ValueError("Every record needs an instance_id")
        chosen = in_calibration(instance_id, args.calibration_fraction, args.seed)
        (calibration if chosen else test).append(row)
    write_jsonl(calibration_path, calibration)
    write_jsonl(test_path, test)
    summary = {
        "calibration": str(calibration_path),
        "calibration_records": len(calibration),
        "test": str(test_path),
        "test_records": len(test),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
