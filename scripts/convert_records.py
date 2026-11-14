"""Map a generic JSON/JSONL dataset file to QCV raw records.

Relative ``--input``, ``--media-root`` and ``--gold-sidecar`` paths that do not
exist in the working directory are looked up in the data directory
(``--data-dir``, ``$QCV_DATA_DIR`` or ``./data``); the output defaults to
``<data>/<task>/<task>_raw.jsonl``. Supported inputs: JSON lines, a JSON list,
a JSON object wrapping a list (``data`` / ``questions`` / ``annotations`` /
``records``), or a JSON object that maps record ids to records (the key is
then available as the field ``__key__``).

    python scripts/convert_records.py --task gqa \\
        --input gqa/val_balanced_questions.json --id-field __key__ \\
        --question-field question --answer-field answer \\
        --media-field imageId --media-suffix .jpg --media-root gqa/images
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qcv.data import default_data_dir
from qcv.utils import read_jsonl, write_jsonl

KEY_FIELD = "__key__"
WRAPPER_KEYS = ("data", "questions", "annotations", "records")


def get_path(value: Any, path: str) -> Any:
    """Follow a dotted path through nested objects and lists."""
    current = value
    for component in path.split("."):
        if isinstance(current, list):
            current = current[int(component)]
        else:
            current = current[component]
    return current


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from read_jsonl(path)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
        return
    if isinstance(payload, dict):
        for key in WRAPPER_KEYS:
            if isinstance(payload.get(key), list):
                yield from payload[key]
                return
        if payload and all(isinstance(value, dict) for value in payload.values()):
            for key, value in payload.items():
                yield {KEY_FIELD: key, **value}
            return
    raise ValueError(
        "Input JSON must be a list, wrap a list under data/questions/annotations/records, "
        "or map record ids to records"
    )


def resolve(path: str, data_dir: Path) -> Path:
    """A path as given if it exists, otherwise the same path inside the data directory."""
    candidate = Path(path)
    if candidate.exists() or candidate.is_absolute():
        return candidate
    return data_dir / candidate


def convert(
    records: Iterable[dict[str, Any]],
    task: str,
    question_field: str,
    answer_field: str,
    media_field: str,
    id_field: str | None = None,
    media_root: Path | None = None,
    media_suffix: str = "",
    media_type: str = "image",
    sidecar: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sidecar = sidecar or {}
    output: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if id_field is None:
            instance_id = str(record.get(KEY_FIELD, f"{task}-{index:07d}"))
        else:
            instance_id = str(get_path(record, id_field))
        media_value = f"{get_path(record, media_field)}{media_suffix}"
        media_path = str(media_root / media_value) if media_root is not None else media_value
        output.append(
            {
                "instance_id": instance_id,
                "task": task,
                "question": str(get_path(record, question_field)),
                "answer": str(get_path(record, answer_field)),
                "media": {"type": media_type, "path": media_path},
                "gold": {str(k): str(v) for k, v in sidecar.get(instance_id, {}).items()},
                "metadata": {"source_record": record},
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="default: <data>/<task>/<task>_raw.jsonl")
    parser.add_argument("--data-dir", help="data directory (default: $QCV_DATA_DIR or ./data)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--id-field", help="default: the record key, else <task>-<index>")
    parser.add_argument("--question-field", required=True)
    parser.add_argument("--answer-field", required=True)
    parser.add_argument("--media-field", required=True)
    parser.add_argument("--media-root", default="")
    parser.add_argument("--media-suffix", default="", help="appended to the media field value")
    parser.add_argument("--media-type", choices=["image", "video"], default="image")
    parser.add_argument(
        "--gold-sidecar", help="JSON object mapping instance IDs to variable-answer maps"
    )
    args = parser.parse_args()

    data_dir = default_data_dir(args.data_dir)
    sidecar: dict[str, dict[str, Any]] = {}
    if args.gold_sidecar:
        raw_sidecar = json.loads(resolve(args.gold_sidecar, data_dir).read_text(encoding="utf-8"))
        sidecar = {str(key): dict(value) for key, value in raw_sidecar.items()}
    rows = convert(
        load_records(resolve(args.input, data_dir)),
        task=args.task,
        question_field=args.question_field,
        answer_field=args.answer_field,
        media_field=args.media_field,
        id_field=args.id_field,
        media_root=resolve(args.media_root, data_dir) if args.media_root else None,
        media_suffix=args.media_suffix,
        media_type=args.media_type,
        sidecar=sidecar,
    )
    output = Path(args.output or data_dir / args.task / f"{args.task}_raw.jsonl")
    write_jsonl(output, rows)
    print(json.dumps({"written": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
