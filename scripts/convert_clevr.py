"""Convert the official CLEVR question/scene JSON files to QCV raw records.

Gold answers of the auxiliary variables (needed for calibration and for the
analysis metrics, never at test time) are derived from the scene graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qcv.compilers import GrammarCompiler, default_grammar_directory, load_task_grammar
from qcv.data import default_data_dir
from qcv.data.download import ensure_file
from qcv.utils import write_jsonl


def execute_program(scene: dict[str, Any], program: list[dict[str, Any]]) -> list[Any]:
    objects = scene["objects"]
    relationships = scene.get("relationships", {})
    outputs: list[Any] = []
    for node in program:
        function = str(node["function"])
        inputs = [outputs[int(index)] for index in node.get("inputs", [])]
        values = list(node.get("value_inputs", []))
        if function == "scene":
            result: Any = set(range(len(objects)))
        elif function.startswith("filter_"):
            attribute = function.removeprefix("filter_")
            expected = values[0]
            result = {index for index in inputs[0] if objects[index][attribute] == expected}
        elif function == "unique":
            if len(inputs[0]) != 1:
                raise ValueError("unique received a non-singleton set")
            result = next(iter(inputs[0]))
        elif function == "relate":
            relation = str(values[0])
            anchor = int(inputs[0])
            relation_rows = relationships[relation]
            result = {int(index) for index in relation_rows[anchor]}
        elif function.startswith("same_"):
            attribute = function.removeprefix("same_")
            anchor = int(inputs[0])
            expected = objects[anchor][attribute]
            result = {
                index
                for index, obj in enumerate(objects)
                if index != anchor and obj[attribute] == expected
            }
        elif function == "union":
            result = set(inputs[0]).union(inputs[1])
        elif function == "intersect":
            result = set(inputs[0]).intersection(inputs[1])
        elif function == "count":
            result = len(inputs[0])
        elif function == "exist":
            result = len(inputs[0]) > 0
        elif function.startswith("query_"):
            attribute = function.removeprefix("query_")
            result = objects[int(inputs[0])][attribute]
        elif function.startswith("equal_"):
            result = inputs[0] == inputs[1]
        elif function == "less_than":
            result = int(inputs[0]) < int(inputs[1])
        elif function == "greater_than":
            result = int(inputs[0]) > int(inputs[1])
        else:
            raise ValueError(f"Unsupported CLEVR program function {function!r}")
        outputs.append(result)
    return outputs


def derive_gold(
    compiled: Any,
    question: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, str]:
    answer = str(question["answer"]).lower()
    rule = compiled.rule_name
    if rule == "count_complement":
        count_a = int(answer)
        total = len(scene["objects"])
        complement = total - count_a
        return {
            "count_a": str(count_a),
            "exists_a": "yes" if count_a > 0 else "no",
            "count_not_a": str(complement),
            "count_total": str(total),
            "exists_not_a": "yes" if complement > 0 else "no",
        }
    if rule == "existence_count":
        outputs = execute_program(scene, question["program"])
        final_node = question["program"][-1]
        if final_node["function"] != "exist" or not final_node.get("inputs"):
            raise ValueError("Existence question does not end in an executable exist node")
        object_set = outputs[int(final_node["inputs"][0])]
        count_a = len(object_set)
        total = len(scene["objects"])
        complement = total - count_a
        return {
            "exists_a": "yes" if bool(outputs[-1]) else "no",
            "count_a": str(count_a),
            "count_not_a": str(complement),
            "count_total": str(total),
            "exists_not_a": "yes" if complement > 0 else "no",
        }
    if rule == "spatial_inverse_categorical":
        inverse = {"left": "right", "right": "left", "front": "behind", "behind": "front"}
        if answer not in inverse:
            raise ValueError(f"Unsupported spatial relation answer {answer!r}")
        return {
            "relation_ab": answer,
            "relation_ba": inverse[answer],
            "is_left": "yes" if answer == "left" else "no",
            "is_right": "yes" if answer == "right" else "no",
            "is_front": "yes" if answer == "front" else "no",
            "is_behind": "yes" if answer == "behind" else "no",
        }
    raise ValueError(f"Gold derivation is not defined for compiler rule {rule!r}")


def convert(
    questions_path: Path,
    scenes_path: Path,
    images_dir: Path,
    split: str,
    grammar: Path,
    drop_incomplete_gold: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Compile every question of one CLEVR split and attach scene-derived gold answers.

    Returns the records and the counts of questions read, compiled and written.
    """
    question_payload = json.loads(questions_path.read_text(encoding="utf-8"))
    scene_payload = json.loads(scenes_path.read_text(encoding="utf-8"))
    scenes = {scene["image_filename"]: scene for scene in scene_payload["scenes"]}
    compiler = GrammarCompiler(load_task_grammar(grammar))
    rows: list[dict[str, Any]] = []
    compiled = 0
    for position, question in enumerate(question_payload["questions"]):
        image_filename = str(question["image_filename"])
        instance_id = f"clevr-{split}-{int(question.get('question_index', position)):06d}"
        result = compiler.compile(instance_id, str(question["question"]))
        if result.compiled is None:
            continue
        compiled += 1
        try:
            gold = derive_gold(result.compiled, question, scenes[image_filename])
        except (KeyError, ValueError, TypeError):
            if drop_incomplete_gold:
                continue
            gold = {result.compiled.target.variable: str(question["answer"]).lower()}
        rows.append(
            {
                "instance_id": instance_id,
                "task": "clevr",
                "question": str(question["question"]),
                "answer": str(question["answer"]).lower(),
                "media": {"type": "image", "path": str(images_dir / image_filename)},
                "gold": gold,
                "metadata": {
                    "image_filename": image_filename,
                    "question_family_index": question.get("question_family_index"),
                    "split": question_payload.get("info", {}).get("split", split),
                },
            }
        )
    counts = {"questions": len(question_payload["questions"]), "compiled": compiled}
    return rows, {**counts, "written": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--data-dir", help="data directory (default: $QCV_DATA_DIR or ./data)")
    parser.add_argument("--questions", help="default: <data>/clevr/CLEVR_v1.0/questions/...")
    parser.add_argument("--scenes", help="default: <data>/clevr/CLEVR_v1.0/scenes/...")
    parser.add_argument("--images-dir", help="default: <data>/clevr/CLEVR_v1.0/images/<split>")
    parser.add_argument("--output", help="default: <data>/clevr/clevr_<split>_raw.jsonl")
    parser.add_argument("--grammar", default=str(default_grammar_directory() / "clevr.yaml"))
    parser.add_argument("--drop-incomplete-gold", action="store_true")
    args = parser.parse_args()

    data_dir = default_data_dir(args.data_dir)
    root = data_dir / "clevr" / "CLEVR_v1.0"
    questions = Path(args.questions or root / "questions" / f"CLEVR_{args.split}_questions.json")
    scenes = Path(args.scenes or root / "scenes" / f"CLEVR_{args.split}_scenes.json")
    images_dir = Path(args.images_dir or root / "images" / args.split)
    output = Path(args.output or data_dir / "clevr" / f"clevr_{args.split}_raw.jsonl")
    # Files under the default layout are fetched (questions and scenes only) when absent.
    if args.questions is None:
        ensure_file(questions, "clevr", data_dir)
    if args.scenes is None:
        ensure_file(scenes, "clevr", data_dir)

    rows, counts = convert(
        questions, scenes, images_dir, args.split, Path(args.grammar), args.drop_incomplete_gold
    )
    write_jsonl(output, rows)
    print(json.dumps({**counts, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
