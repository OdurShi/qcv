"""Build the CLEVR records shipped in ``data/clevr/`` from the official CLEVR v1.0 files.

The script runs the two commands below and then writes ``MANIFEST.json`` and
the licence and attribution files next to the records:

    python scripts/convert_clevr.py --split val --drop-incomplete-gold
    python scripts/split_records.py --input data/clevr/clevr_val_raw.jsonl \\
        --calibration-output data/clevr/clevr_val_calibration.jsonl.gz \\
        --test-output data/clevr/clevr_val_test.jsonl.gz \\
        --calibration-fraction 0.2 --seed 0

``convert_clevr.py`` keeps the questions of the split that the CLEVR grammar
compiles and derives the gold answers of the auxiliary variables from the
scene graphs; ``split_records.py`` assigns each record to the calibration or
the test file by a hash of the seed and the instance id. The CLEVR annotation
archive is fetched into ``data/clevr/`` first if it is not there.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from qcv.compilers import CompilerRegistry, default_grammar_directory
from qcv.data import default_data_dir
from qcv.data.download import RECORD_NAME, sha256_of
from qcv.data.sources import SOURCES
from qcv.utils import json_dump, read_jsonl

SCRIPTS = Path(__file__).resolve().parent
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = """# CLEVR attribution

The files `clevr_{split}_calibration.jsonl.gz` and `clevr_{split}_test.jsonl.gz` in this
directory are derived from CLEVR v1.0.

* Dataset: CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual
  Reasoning, v1.0 (J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. L. Zitnick,
  R. Girshick, 2017)
* Source: {homepage}
* Copyright notice of the dataset: "{copyright}"
* Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), {licence_url};
  the licence text is in `LICENSE.txt`.

Changes made: the questions of the {split} split that the grammar `qcv/grammars/clevr.yaml`
compiles were selected ({records} of {questions}); each was converted to one JSON line with the
question, its answer, the image file name, and answers of related questions (object counts and
existence) computed from the CLEVR scene graphs and functional programs; the records were
divided into two files. Images are not included.
"""


def _run(script: str, *arguments: str) -> dict:
    """Run one of the repository scripts and return the JSON summary it prints."""
    command = [sys.executable, str(SCRIPTS / script), *arguments]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


def _describe(path: Path) -> dict:
    """Record count, size, checksum and per-rule counts of one shipped record file."""
    compiler = CompilerRegistry.from_directory().compiler_for("clevr")
    rules: Counter[str] = Counter()
    records = 0
    for row in read_jsonl(path):
        records += 1
        rules[compiler.compile(row["instance_id"], row["question"]).compiled.rule_name] += 1
    with gzip.open(path, "rb") as handle:
        content_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return {
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "uncompressed_sha256": content_sha256,
        "rules": dict(sorted(rules.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="data directory (default: $QCV_DATA_DIR or ./data)")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = default_data_dir(args.data_dir)
    target = data_dir / "clevr"
    raw = target / f"clevr_{args.split}_raw.jsonl"
    calibration = target / f"clevr_{args.split}_calibration.jsonl.gz"
    test = target / f"clevr_{args.split}_test.jsonl.gz"

    converted = _run(
        "convert_clevr.py", "--data-dir", str(data_dir), "--split", args.split,
        "--drop-incomplete-gold",
    )  # fmt: skip
    _run(
        "split_records.py", "--input", str(raw),
        "--calibration-output", str(calibration), "--test-output", str(test),
        "--calibration-fraction", str(args.calibration_fraction), "--seed", str(args.seed),
    )  # fmt: skip

    source = SOURCES["clevr"]
    upstream = target / "CLEVR_v1.0"
    if (upstream / "LICENSE.txt").is_file():
        shutil.copyfile(upstream / "LICENSE.txt", target / "LICENSE.txt")
    copyright_file = upstream / "COPYRIGHT.txt"
    copyright_notice = "CLEVR (c) 2017, Facebook, Inc."
    if copyright_file.is_file():
        copyright_notice = (
            copyright_file.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        )
    (target / "ATTRIBUTION.md").write_text(
        ATTRIBUTION.format(
            split=args.split,
            homepage=source.homepage,
            copyright=copyright_notice,
            licence_url=LICENCE_URL,
            records=converted["written"],
            questions=converted["questions"],
        ),
        encoding="utf-8",
    )

    archive = {"url": source.files[0].url}
    record_path = target / RECORD_NAME
    if record_path.is_file():
        fetched = json.loads(record_path.read_text(encoding="utf-8"))["files"]
        entry = fetched.get(source.files[0].name, {})
        archive.update({key: entry[key] for key in ("bytes", "sha256") if key in entry})
    grammar = default_grammar_directory() / "clevr.yaml"
    manifest = {
        "dataset": f"CLEVR v1.0 {args.split} questions compiled by the CLEVR grammar",
        "source": {"homepage": source.homepage, "archive": archive},
        "licence": "CC BY 4.0",
        "licence_url": LICENCE_URL,
        "attribution": copyright_notice,
        "grammar": {"file": "qcv/grammars/clevr.yaml", "sha256": sha256_of(grammar)},
        "selection": {
            "split": args.split,
            "questions": converted["questions"],
            "compiled": converted["compiled"],
            "records": converted["written"],
        },
        "partition": {
            "method": "hash of seed and instance_id (scripts/split_records.py)",
            "calibration_fraction": args.calibration_fraction,
            "seed": args.seed,
        },
        "commands": [
            f"python scripts/download_data.py clevr --data-dir {data_dir}",
            f"python scripts/build_clevr_slice.py --data-dir {data_dir} --split {args.split} "
            f"--calibration-fraction {args.calibration_fraction} --seed {args.seed}",
        ],
        "files": {calibration.name: _describe(calibration), test.name: _describe(test)},
    }
    json_dump(target / "MANIFEST.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
