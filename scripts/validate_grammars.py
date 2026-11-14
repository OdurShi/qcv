"""Compile one example question per rule of every shipped grammar and report code sizes."""

from __future__ import annotations

import json

from qcv.codebook import build_codebook
from qcv.compilers import CompilerRegistry

CASES = {
    "clevr": [
        "How many red cubes are there?",
        "Are there any red cubes?",
        "What is the spatial relation of cube a relative to cube b?",
        "Is cube a left of cube b?",
    ],
    "gqa": [
        "Are there any red cars?",
        "How many red cars are there?",
        "What is the spatial relation of car a relative to car b?",
    ],
    "chartqa": [
        "How many more bicycles than motorcycles are there?",
        "What is the signed difference between series a and series b",
        "What is the sum of series a and series b",
        "Is series a greater than series b",
    ],
    "agqa": [
        "How many times did the person open the door?",
        "Did the person open the door happen",
        "Did event a happen before event b",
        "What is the temporal relation of event a to event b",
    ],
    "synthetic": [
        "How many more red cubes than blue spheres are there?",
        "What is the spatial relation of the red cube relative to the blue sphere?",
    ],
}


def validate() -> list[dict[str, object]]:
    registry = CompilerRegistry.from_directory()
    results: list[dict[str, object]] = []
    for task, questions in CASES.items():
        compiler = registry.compiler_for(task)
        for index, question in enumerate(questions):
            result = compiler.compile(f"{task}-{index}", question)
            if result.compiled is None:
                raise RuntimeError(
                    f"Grammar validation failed for {task!r}: {question!r} -> {result.reason}"
                )
            codebook = build_codebook(result.compiled, result.compiled.all_questions)
            if codebook.size == 0:
                raise RuntimeError(
                    f"Empty feasible codebook for {task!r}: {result.compiled.rule_name}"
                )
            results.append(
                {
                    "task": task,
                    "rule": result.compiled.rule_name,
                    "question": question,
                    "auxiliaries": len(result.compiled.auxiliaries),
                    "codewords": codebook.size,
                    "target_classes": len(codebook.target_classes),
                    "dimension": codebook.dimension,
                }
            )
    return results


def main() -> None:
    results = validate()
    print(json.dumps({"validated": len(results), "cases": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
