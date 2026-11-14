"""Command line interface: ``qcv <command>``.

compile / score / calibrate / infer / evaluate / compare form the paper's
pipeline; ``baseline`` adds the score-file baselines; ``synth`` and ``demo``
provide a dataset-free, VLM-free synthetic path through the same commands.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from .baselines import BASELINE_METHODS
from .comparison import compare_prediction_files
from .demo import run_demo
from .inference import InferenceOptions
from .pipeline import (
    baseline_jsonl,
    calibrate_jsonl,
    compile_jsonl,
    evaluate_jsonl,
    infer_jsonl,
    score_jsonl_hf,
    score_jsonl_synthetic,
    synthesize_raw_jsonl,
)
from .selection import SELECTION_METHODS
from .synthetic import SyntheticConfig
from .types import METRIC_MODES


def _print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _add_simulator_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = SyntheticConfig()
    group = parser.add_argument_group("synthetic simulator")
    group.add_argument(
        "--error-rate",
        type=float,
        default=defaults.error_rate,
        help="probability that the simulated percept of a question is wrong",
    )
    group.add_argument(
        "--correlation",
        type=float,
        default=defaults.correlation,
        help="probability that a restatement of the target repeats the target's percept",
    )
    group.add_argument("--seed", type=int, default=defaults.seed)


def _simulator_config(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(error_rate=args.error_rate, correlation=args.correlation, seed=args.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qcv", description="Question-Code Verification pipeline")
    parser.add_argument("--verbose", action="store_true", help="log progress to stderr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub = subparsers.add_parser("compile", help="compile target questions into typed QCV pools")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--grammars", help="grammar directory (default: grammars shipped with qcv)")
    sub.add_argument("--drop-uncovered", action="store_true")

    sub = subparsers.add_parser("score", help="score complete answer alphabets")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--backend", choices=["hf", "synthetic"], default="hf")
    sub.add_argument("--model", help="Hugging Face model id or local path (hf backend)")
    sub.add_argument("--device", default="cuda")
    sub.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    sub.add_argument("--attention-implementation")
    sub.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow model repositories that ship custom code (off by default)",
    )
    sub.add_argument(
        "--answer-separator",
        choices=["auto", "none", "space"],
        default="auto",
        help="text between the prompt and a candidate answer; auto = none after a chat template",
    )
    sub.add_argument(
        "--max-batch-candidates",
        type=int,
        default=0,
        help="0 scores the complete answer alphabet in one padded batch; "
        "a positive value splits it into chunks to bound memory",
    )
    _add_simulator_arguments(sub)

    sub = subparsers.add_parser("calibrate", help="fit role temperatures and residual covariances")
    sub.add_argument("--input", required=True)
    sub.add_argument("--output-dir", required=True)
    sub.add_argument("--max-budget", type=int, default=9)

    sub = subparsers.add_parser("infer", help="run QCV selection and feasible projection")
    sub.add_argument("--input", required=True)
    sub.add_argument("--artifacts", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--budget", type=int, default=5)
    sub.add_argument("--selection", choices=SELECTION_METHODS, default="qcv")
    sub.add_argument("--metric", choices=METRIC_MODES, default="full")
    sub.add_argument("--non-strict-coverage", action="store_true")
    sub.add_argument("--random-seed", type=int, default=0)
    sub.add_argument("--exact-subset-limit", type=int, default=10_000)

    sub = subparsers.add_parser("baseline", help="run a baseline computable from a score file")
    sub.add_argument("--input", required=True)
    sub.add_argument("--artifacts", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--method", choices=BASELINE_METHODS, required=True)
    sub.add_argument("--budget", type=int, default=5)
    sub.add_argument("--seed", type=int, default=0)

    sub = subparsers.add_parser(
        "evaluate", help="compute accuracy, correction and geometry metrics"
    )
    sub.add_argument("--scores", required=True)
    sub.add_argument("--predictions", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--artifacts")

    sub = subparsers.add_parser("compare", help="paired bootstrap and exact McNemar comparison")
    sub.add_argument("--scores", required=True)
    sub.add_argument("--first", required=True)
    sub.add_argument("--second", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--bootstrap-replicates", type=int, default=10_000)
    sub.add_argument("--seed", type=int, default=0)

    sub = subparsers.add_parser("synth", help="write synthetic raw records (no dataset needed)")
    sub.add_argument("--output-dir", required=True)
    sub.add_argument("--calibration-size", type=int, default=1500)
    sub.add_argument("--test-size", type=int, default=2000)
    _add_simulator_arguments(sub)

    sub = subparsers.add_parser("demo", help="run the whole pipeline on synthetic data")
    sub.add_argument("--output-dir", default="runs/demo")
    sub.add_argument("--calibration-size", type=int, default=1500)
    sub.add_argument("--test-size", type=int, default=2000)
    sub.add_argument("--budget", type=int, default=5)
    sub.add_argument("--bootstrap-replicates", type=int, default=2000)
    sub.add_argument(
        "--data-dir",
        help="directory holding synthetic/ raw records (default: $QCV_DATA_DIR or ./data)",
    )
    _add_simulator_arguments(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "compile":
        result = compile_jsonl(
            args.input, args.output, args.grammars, keep_uncovered=not args.drop_uncovered
        )
    elif args.command == "score":
        if args.backend == "synthetic":
            result = score_jsonl_synthetic(args.input, args.output, _simulator_config(args))
        else:
            if not args.model:
                parser.error("--model is required for --backend hf")
            from .models import ModelLoadOptions

            options = ModelLoadOptions(
                model_name_or_path=args.model,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                attention_implementation=args.attention_implementation,
                max_batch_candidates=args.max_batch_candidates,
                answer_separator=args.answer_separator,
            )
            result = score_jsonl_hf(args.input, args.output, options)
    elif args.command == "calibrate":
        result = calibrate_jsonl(args.input, args.output_dir, args.max_budget)
    elif args.command == "infer":
        result = infer_jsonl(
            args.input,
            args.artifacts,
            args.output,
            InferenceOptions(
                budget=args.budget,
                selection=args.selection,
                metric=args.metric,
                strict_coverage=not args.non_strict_coverage,
                random_seed=args.random_seed,
                exact_subset_limit=args.exact_subset_limit,
            ),
        )
    elif args.command == "baseline":
        result = baseline_jsonl(
            args.input, args.artifacts, args.output, args.method, args.budget, args.seed
        )
    elif args.command == "evaluate":
        result = evaluate_jsonl(
            args.scores, args.predictions, args.output, artifact_directory=args.artifacts
        )
    elif args.command == "compare":
        result = compare_prediction_files(
            args.scores,
            args.first,
            args.second,
            args.output,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    elif args.command == "synth":
        result = synthesize_raw_jsonl(
            args.output_dir, args.calibration_size, args.test_size, _simulator_config(args)
        )
    else:
        print(
            run_demo(
                args.output_dir,
                calibration_size=args.calibration_size,
                test_size=args.test_size,
                budget=args.budget,
                config=_simulator_config(args),
                bootstrap_replicates=args.bootstrap_replicates,
                data_dir=args.data_dir,
            )
        )
        return
    _print_result(result)


if __name__ == "__main__":
    main()
