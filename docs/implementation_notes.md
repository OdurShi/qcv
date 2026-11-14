# Implementation notes

This file maps the paper to the code, records every decision the paper's main
text leaves open, states the scope of the implementation and lists the runs
that were made with it.

The specification used for this repository is the 8-page main text of
"Questions as Codes: Correcting Multimodal Reasoning at Test Time". The main
text defers many details to appendix sections (A.1-A.8, B.1-B.2, C.1-C.3). Where
it does so, the choice made here is listed under
[Unspecified details and choices](#unspecified-details-and-choices) and should be
checked against the appendix.

## Paper-to-code map

| Paper | Content | Code |
|---|---|---|
| Typed Questions and Feasible Codes | compiler: `q0` + task grammar -> ordered pool `P(u)`, roles, alphabets, predicates `g_r` | `qcv/compilers/grammar.py::GrammarCompiler.compile`, `qcv/grammars/*.yaml` |
| | predicates `g_r` | `qcv/constraints.py::evaluate_constraint` |
| Eq. (1) | feasible codebook `C_Q̄`, target classes `C_Q̄(y)`, one-hot blocks | `qcv/codebook.py::build_codebook`, `enumerate_feasible_indices`, `pool_feasible_indices` |
| Eq. (2), first line | length-normalised teacher-forced score `l_k(a)` | `qcv/models/hf.py::HuggingFaceVLMScorer.score_question`, `mean_answer_logprob`, `answer_token_positions` |
| Eq. (2), second line | calibrated distribution `p_k`, temperature `T_tau` fitted by NLL | `qcv/temperature.py::calibrated_probabilities`, `fit_temperature` |
| Eq. (2), third line | stacked observation `z_Q̄` | `qcv/geometry.py::stack_observation` |
| Residual Geometry | residuals `r = z - c` grouped by ordered role signature | `qcv/calibration.py::fit_calibration`, `qcv/signatures.py::signature_for_questions` |
| Eq. (3) | centred covariance `S`, Ledoit-Wolf shrinkage with `1/N` floor, `mu`, `Sigma`, `W = Sigma^{-1/2}` | `qcv/covariance.py::fit_residual_covariance`, `ledoit_wolf_shrinkage`, `inverse_sqrt` |
| Eq. (4) | target-class separation `d_tar` | `qcv/geometry.py::target_class_separation`, `qcv/selection.py::set_separation` |
| Eq. (5) | greedy selection of `B - 1` auxiliaries, ties by template index | `qcv/selection.py::greedy_select` |
| App. A.4 (referenced) | exhaustive-search comparison | `qcv/selection.py::exact_select` |
| Eq. (6) | energy `E(c; z) = ||W (z - c)||^2` | `qcv/decoding.py::codeword_energies` |
| Eq. (7) | target prediction by feasible projection | `qcv/decoding.py::feasible_projection_decode` |
| Eq. (8) | deterministic correction condition, `kappa = 2r / d_tar` | `qcv/geometry.py::whitened_residual_radius`, `normalized_margin`; audit in `qcv/metrics.py::_separation_audit`; property test `tests/test_decoding.py::test_deterministic_correction_condition` |
| Inference Protocol | coverage, exactly `B` blocks, target-only fallback | `qcv/inference.py::InferenceEngine.infer`, `coverage_reason` |
| Table 1 / 3 controls | Fixed related probes, Random-QCV, Reliability-QCV, Diagonal-QCV | `qcv/selection.py::fixed_select`, `random_select`, `reliability_select`; `qcv/geometry.py::whitening_for_mode` |
| Table 1 | Target-only | `qcv/baselines.py::target_only_prediction` |
| Table 1 / 3, Fig. 2 | Repeated target (score-file stand-in, see below) | `qcv/baselines.py::repeated_target_prediction` |
| Table 3 columns | `Acc_all`, `Cov.`, `Acc_cov`, `Corr.`, `Corrupt.`, `rho_B`, `B_eff`, `S_B`, `Err. AUROC` | `qcv/metrics.py::evaluate_predictions` and the functions it calls |
| Table 1 `Avg.` | macro average over benchmarks | `macro_overall_accuracy` in `evaluate_predictions`; `scripts/make_table.py` |
| Q4, Fig. 4 | `kappa` audit | `separation_audit` in `evaluate_predictions` |
| Q4, Fig. 5 | budget sweep, Spearman correlations | `scripts/run_ablation_sweep.py`, `configs/budget_sweep.yaml` |
| Statistical tests (App. B.2, referenced) | paired comparison | `qcv/metrics.py::paired_bootstrap_accuracy_difference`, `mcnemar_exact`, `holm_bonferroni`; `qcv/comparison.py` |

## Unspecified details and choices

### Compiler and feasible codes

1. **Feasible code of a sub-pool.** Eq. (1) defines `C_Q̄` through predicates
   `g_r(a) = 1` over the answer tuple of the selected set `Q̄`, while the
   predicates are emitted for the whole pool. A tuple over a subset is treated
   as feasible iff it extends to a jointly valid tuple over the whole pool
   (projection of the pool-level feasible set, i.e. unselected variables are
   eliminated). This is what the pipeline figure of the paper shows for the
   candidate sets of its counting example (`d = b - m`, `t = b + m` give
   `d = t - 2m` on `{d, m, t}`). The original code instead dropped every
   predicate that mentioned an unselected variable, which left such sets
   unconstrained.
2. **Grammars.** `qcv/grammars/{clevr,gqa,chartqa,agqa}.yaml` are compact
   reference grammars with four or five auxiliary templates per rule and small
   integer alphabets. The paper's full grammar (App. A.1) is not reproduced
   here. Consequences: budgets `B >= 7` need at least six auxiliaries and are
   therefore outside coverage with the shipped benchmark grammars, and real
   ChartQA values need alphabets frozen from calibration data
   (`scripts/fit_alphabets.py`).
3. **Compilation.** A question compiles iff exactly one rule pattern matches
   its normalised text (lower-case, collapsed whitespace, no trailing
   punctuation). Auxiliaries are ordered by template index, then text;
   duplicates of the target or of an earlier auxiliary are dropped; an instance
   whose pool-level code is empty is rejected. Questions that do not compile
   have no typed answer alphabet, so they cannot be scored and are excluded at
   `qcv score` (counted as `skipped`); coverage is reported relative to compiled
   examples.

### Scoring (Eq. 2)

4. **Prompt.** The user turn is
   `"{question}\nReturn exactly one canonical answer from this finite set: [a1, a2, ...].\nAnswer:"`
   with the image or video placed first, rendered through the processor's chat
   template when one exists, otherwise prefixed with `<image>` / `<video>`.
   The paper's prompt (App. A.2) is not available in the main text.
5. **Answer tokens.** `L(a)` counts the tokens of the canonical answer string.
   After a chat template the answer follows the assistant header directly; after
   a plain prompt it follows `"Answer:"` and one space. `--answer-separator
   space` restores the original code's behaviour (a leading space in both
   cases). Answer tokens are located as the last occurrence of the separately
   tokenised answer among the non-padding tokens, which works for left and right
   padding; a mismatch raises an error instead of scoring wrong positions.
6. **Batching and reuse.** One call scores the whole alphabet in one padded
   batch (optionally chunked). The image is passed once per candidate; the
   paper's statement that the visual encoding is computed once and reused across
   the `B` evaluations is **not** implemented, so FLOPs and latency of this
   scorer are not comparable with the paper's Table 2.
7. **Whole pool is scored.** `qcv score` stores scores for the target and every
   auxiliary of the pool so that one file serves all budgets and selection
   variants. Inference reads exactly `B` blocks; the decision is identical to
   scoring only the selected questions, but the scoring cost is not.
8. **Remote code.** `trust_remote_code` is off unless `--trust-remote-code` is
   given (the original code enabled it unconditionally).

### Calibration (Eqs. 2-3)

9. **Temperature.** One scalar per role, where a role is identified by its
   name and its alphabet. Fitted by mean categorical NLL with L-BFGS-B over
   `log T` in `[-12, 12]`, started at `T = 1`.
10. **Signature.** The ordered role signature of a set is the sequence of role
    keys with the target first and auxiliaries in template order. One
    covariance is fitted per signature and per subset size, for every subset
    with at most `max_budget - 1` auxiliaries (subset covariances are fitted
    separately, they are not sub-blocks of a pool covariance after shrinkage).
11. **Covariance.** `S` uses the `1/N` normalisation and one global mean per
    signature. `alpha_LW` is the Ledoit-Wolf (2004) estimate
    `min(b^2, d^2) / d^2` (it equals scikit-learn's `ledoit_wolf` on the same
    data, tested); `alpha = min(max(alpha_LW, 1/N), 1)`. A signature is
    unsupported when it has fewer than two residuals, when `mu <= 1e-12`
    ("zero residual variance") or when the smallest eigenvalue of `Sigma` is
    `<= 1e-12`.
12. **Calibration records.** A record contributes to a signature only if gold
    answers exist for all its questions and the gold tuple is a feasible
    codeword of that set.

### Selection (Eqs. 4-5)

13. **Greedy.** Step `t` evaluates `d_tar({q0} ∪ Q_t ∪ {q})` with the whitening
    transform of that enlarged set. Candidates are scanned in compiler order and
    replace the incumbent only on a strict improvement larger than a relative
    tolerance of `1e-12`, so numerically tied candidates resolve to the lowest
    template index.
14. **Exhaustive search.** Ties go to the lexicographically first subset in
    compiler order; above 10,000 candidate subsets it falls back to greedy.
15. **Controls.** Fixed related probes = the first `B - 1` templates.
    Random-QCV = a uniform subset seeded by `(seed, instance id)`.
    Reliability-QCV = the `B - 1` auxiliaries whose roles have the highest
    arg-max accuracy on the calibration split (ties by template index).
    Diagonal-QCV = `W = diag(Sigma)^{-1/2}` for both selection and decoding
    (a fully diagonal metric; a block-diagonal variant that keeps
    within-question covariance is another possible reading of the paper).
    `identity` (`W = I`) is an extra diagnostic that is not in the paper.

### Decoding (Eqs. 6-7)

16. **Ties.** Class ties and codeword ties go to the first entry in canonical
    alphabet / answer-product order; energies within a relative `1e-12` count
    as tied. Projected codebooks contain each codeword once.
17. **Analysis quantities.** `target_margin` = runner-up class energy minus
    winning class energy. `projection_gap` = winning feasible energy minus the
    minimum energy over all one-hot tuples of the selected set (computed exactly
    on a grid; skipped above 250,000 tuples). The paper's "projection gap used
    for analysis" (App. A.5) is not defined in the main text; this is the
    definition of the original code.

### Inference protocol

18. **Coverage.** A record is covered when it has at least `B - 1` auxiliaries,
    a temperature for every pool question, and a positive-definite covariance
    for every required signature whose code has at least two feasible target
    classes. Required signatures are: for greedy QCV, Diagonal-QCV and
    exhaustive search, all sets with `1..B-1` auxiliaries (`{q0}` when
    `B = 1`), so that selection is well defined whichever path it takes; for
    fixed, random and reliability selection, only the selected set. The paper's
    Table 3 reports different coverage for the fixed and random controls than
    for QCV, which a single rule for all methods (as in the original code)
    cannot produce. `--non-strict-coverage` lets greedy skip unsupported
    candidates instead.
19. **Fallback.** `argmax_y p_0[y]`, computed from the raw target scores
    (temperature scaling preserves the arg-max), first alphabet entry on ties.

### Metrics

20. **Correction / corruption.** `P(final right | target-only wrong)` and
    `P(final wrong | target-only right)` over all evaluated examples, fallback
    examples included (this matches `Acc_all = Acc_0 (1 - Corrupt) + (1 - Acc_0)
    Corr` in the paper's Table 3).
21. **`rho_bar_B`.** Each covered example gives a 0/1 error indicator for its
    `B` measurements (target first, then the selected auxiliaries in template
    order; a measurement is the arg-max answer of the calibrated block).
    `rho_bar_B` is the mean Pearson correlation over pairs of measurement slots
    across examples; pairs with a constant slot are skipped.
22. **`B_eff`.** `B / (1 + (B - 1) rho_bar_B)`, capped at `B` when `rho_bar_B`
    is negative. The formula is inferred from the numbers in the paper's text
    (for example `B = 3, rho = 0.68 -> 1.28` and `B = 9, rho = 0.046 -> 6.56`)
    and from the `B_eff` guide lines of its Fig. 5.
23. **`S_B`.** Over examples whose target measurement is wrong, the mean
    fraction of the other `B - 1` measurements that are correct.
24. **Error AUROC.** AUROC of `-target_margin` (and separately of
    `projection_gap`) for detecting wrong final answers, on covered examples
    only. Which score the paper uses, and how uncovered examples enter, is not
    stated in the main text.
25. **Aggregation.** `qcv evaluate` reports pooled metrics, per-task metrics and
    the macro average of per-task accuracy. Geometry statistics that the paper
    macro-averages "across configurations" are obtained by evaluating one file
    per configuration and averaging (`scripts/make_table.py`).
26. **ChartQA.** Numeric answers use the benchmark's usual 5% relaxed match;
    all other answers use normalised exact match.
27. **Separation audit.** With gold answers for the selected auxiliaries,
    `kappa = 2 ||W (z - c*)||_2 / d_tar` is computed per covered example and
    summarised in the bins `kappa < 0.8`, `0.8 <= kappa < 1.2`, `kappa >= 1.2`
    used in the paper's text, together with the number of wrong answers at
    `kappa < 1` (Eq. 8 guarantees zero).
28. **Statistical tests.** Paired percentile bootstrap (10,000 resamples, 95%
    interval) of the accuracy difference, two-sided exact McNemar test, Holm
    adjustment across the methods compared at one budget, and Spearman
    correlation across sweep cells. The paper's tests (App. B.2) are not named
    in the main text.

### Baselines

29. **Repeated target.** With deterministic teacher-forced scores, re-evaluating
    the target gives identical answers, so the paper's repeated-target protocol
    must involve some randomness that the main text does not describe. The
    stand-in here draws `B` answers i.i.d. from the calibrated target
    distribution and takes a plurality vote (ties to the more probable answer).
    It exists to compute the failure geometry of spending the budget on the
    target (`rho_bar_B`, `B_eff`, `S_B`); in expectation it cannot beat the
    arg-max and is not a reproduction of the paper's row.
30. **Not implemented.** Self-Consistency, Best-of-N, beam search, MCTS,
    Least-to-Most, visual question decomposition and implication projection
    need free-form generation or other systems and cannot be derived from
    score files.

## Synthetic simulator

`qcv/synthetic.py` replaces the dataset and the VLM for the quick start and the
end-to-end tests. Scenes are jointly valid answer tuples of the `synthetic`
grammar. Each question is answered from a percept that is wrong with
probability `error_rate`; auxiliaries that restate the target copy the target's
percept with probability `correlation`. The simulator is a test bed for the
mechanics of the method. Results obtained with it are not evidence about real
VLMs or benchmarks.

## Data, splits and figures

* **Benchmarks.** CLEVR, GQA, ChartQA and AGQA-Decomp are fetched by
  `scripts/download_data.py` into `data/<dataset>/` (sources, licences and
  sizes in `data/README.md`, registry in `qcv/data/sources.py`). The source
  URLs were checked with header-only requests on 2026-11-13 (status 200 and the
  sizes listed in the registry). Only the CLEVR annotation archive
  (`CLEVR_v1.0_no_images.zip`, 89,363,837 bytes, SHA-256 `d4435dce...74c4`) was
  fetched while preparing the repository; the other archives were not, so their
  SHA-256 values are recorded by the script at download time (in
  `data/<dataset>/DOWNLOAD.json`) rather than pinned in advance.
* **CLEVR records.** `data/clevr/clevr_val_{calibration,test}.jsonl.gz` are
  built by `scripts/build_clevr_slice.py`: the 15,723 of 149,991 CLEVR v1.0
  validation questions that `qcv/grammars/clevr.yaml` compiles (14,031
  existence and 1,692 counting questions), with gold answers of the auxiliary
  variables derived from the scene graphs and programs. For every record the
  derived target answer equals the CLEVR answer and the gold tuple is a
  feasible codeword (checked by a test). The partition into 3,148 calibration
  and 12,575 test records is the hash split of `scripts/split_records.py`
  (20%, seed 0). Neither the selection nor the partition is the paper's.
* **CLEVR grammar patterns.** Running the converter on the real validation
  questions showed two patterns of the original grammar matching unrelated
  question families: "Are there an equal number of X and Y?" was compiled as
  an existence question (1,173 questions) and comparison or attribute-equality
  questions that mention "left of" were compiled as "is A left of B"
  (2,015 questions). Both patterns now exclude these forms; before the change
  the converter dropped those questions because no gold could be derived.
* **Evaluation slices and splits.** The paper evaluates on fixed,
  compiler-supported slices with a held-out calibration split and defers their
  definition to its appendix. In this repository the slice of a benchmark is
  the set of questions matched by its grammar (`qcv compile`), and
  `scripts/split_records.py` assigns records to calibration or test by a hash
  of a seed and the instance id (default 20% calibration, seed 0). These are
  repository defaults, not the paper's split files.
* **Converters.** `scripts/convert_clevr.py` reads the official CLEVR files and
  derives the gold answers of the auxiliary variables from the scene graphs
  (run on the CLEVR v1.0 validation files, and tested on a hand-made scene in
  the CLEVR format). GQA, ChartQA and AGQA-Decomp go through the generic
  `scripts/convert_records.py`; the gold answers of their auxiliary variables
  have to be supplied as a side file.
* **Downloads and TLS.** The downloader verifies certificates with the
  operating-system trust store (`truststore`) or the `certifi` bundle when
  installed; if Python has no usable CA bundle and `curl` is available, the
  file is fetched with `curl`. Verification is never disabled.
* **Backbone identifiers.** The paper names the backbones (Qwen2.5-VL,
  LLaVA-OneVision, InternVL3); the Hugging Face identifiers in the README are
  this repository's mapping of those names.
* **Synthetic records.** `data/synthetic/` is the output of
  `qcv synth --output-dir data/synthetic` (seed 0). A test regenerates the
  records and compares them byte for byte with the shipped files.
* **Paper figures.** `assets/paper_figures/fig01.png` to `fig05.png` are crops
  of the manuscript PDF at 200 dpi made by `scripts/extract_paper_figures.py`
  (crop boxes are listed in the script; captions and page furniture are
  excluded).

## Scope of the implementation

* Provided: the QCV pipeline, its selection and decoding controls (Fixed
  related probes, Random-QCV, Reliability-QCV, Diagonal-QCV), target-only
  decoding, the metrics of Table 3, the `kappa` audit, budget sweeps and paired
  tests.
* Baselines that need free-form generation or other systems (Self-Consistency,
  Best-of-N, beam search, MCTS, Least-to-Most, visual question decomposition,
  implication projection) are outside the code; `repeated-target` is the
  score-file stand-in described above.
* Table 2 (FLOPs, latency): there is no FLOP counter and the scorer does not
  reuse the visual encoding across evaluations. Per-question wall-clock latency
  is stored in the score-file metadata.
* Figures: the sweep outputs contain the plotted quantities (`sweep.csv`,
  `correlations.json`, `separation_audit`); plotting scripts are not included.
* The appendix-only MathVista-Struct transfer study is outside the code.
* Codebooks, separations and unconstrained-energy grids are cached in memory
  per compiled structure. Grammars whose constraint parameters depend on the
  question text create one structure per instance and make these caches grow.

## Record of runs

All runs below were made on 2026-11-13 on an Apple-silicon laptop CPU with
Python 3.11.18 and the versions in `requirements-dev.lock` (plus `torch 2.15.0`
and `transformers 5.23.0` for the scorer tests). They use the synthetic scenes
and a simulated VLM; no VLM was run on benchmark images and no pretrained model
was involved, so they are not runs of the paper's experiments. Raw outputs are
in `results/`.

| Command | Outcome |
|---|---|
| `ruff check .`, `ruff format --check .` | no findings |
| `pytest` | 264 passed with `torch` and `transformers` installed; 250 passed, 2 skipped (the two scorer test modules) in fresh Python 3.10 and 3.12 environments installed with `pip install ".[dev]"` only |
| `python scripts/download_data.py clevr`, `python scripts/build_clevr_slice.py` | CLEVR annotation archive fetched (89 MB); `data/clevr/` records and `MANIFEST.json` written (15,723 records) |
| `python scripts/validate_grammars.py` | 17 example questions (every rule of the five grammars) compile to non-empty codes with at least two target classes |
| `qcv demo --output-dir runs/demo` | `results/synthetic_demo/default/report.txt` (shown in the README quick start) |
| `qcv demo` with `--correlation 0.0`, `--error-rate 0.3`, and both | `results/synthetic_demo/{independent,hard,hard_independent}/report.txt` |
| `qcv calibrate --max-budget 9` and `scripts/run_ablation_sweep.py --config configs/budget_sweep.yaml` on the default demo data | `results/synthetic_budget_sweep/` |

Synthetic demo under four simulator settings (accuracy in %, 2000 test
examples, `B = 5`):

| `--error-rate` | `--correlation` | target-only | identity-qcv | diagonal-qcv | qcv | qcv - diagonal-qcv (95% CI) |
|---|---|---|---|---|---|---|
| 0.1 (default) | 0.8 (default) | 89.8 | 93.5 | 93.3 | 96.9 | +3.6 (+2.4 to +4.8) |
| 0.1 | 0.0 | 89.8 | 98.9 | 99.0 | 97.8 | -1.2 (-1.8 to -0.5) |
| 0.3 | 0.8 | 70.1 | 75.3 | 75.1 | 76.9 | +1.8 (-0.5 to +4.0) |
| 0.3 | 0.0 | 70.1 | 89.0 | 87.8 | 76.2 | -11.6 (-13.6 to -9.6) |

In all four runs every decoded answer belongs to a feasible codeword and no
example with `kappa < 1` is wrong, as Eq. (8) requires.

Budget sweep on the default demo data (accuracy in % / coverage in %):

| Method | B = 3 | B = 5 | B = 7 | B = 9 |
|---|---|---|---|---|
| target-only | 89.8 / 100 | 89.8 / 100 | 89.8 / 100 | 89.8 / 100 |
| repeated-target | 86.0 / 100 | 87.4 / 100 | 87.8 / 100 | 88.3 / 100 |
| fixed-probes | 90.0 / 100 | 96.8 / 100 | 97.0 / 100 | 96.2 / 75.1 |
| random-qcv | 90.6 / 100 | 94.1 / 100 | 96.1 / 100 | 96.2 / 75.1 |
| reliability-qcv | 88.3 / 100 | 91.5 / 100 | 97.2 / 100 | 96.2 / 75.1 |
| diagonal-qcv | 92.3 / 100 | 93.3 / 100 | 96.1 / 100 | 96.2 / 75.1 |
| qcv | 93.2 / 100 | 96.9 / 100 | 97.2 / 100 | 96.2 / 75.1 |

At `B = 9` the spatial rule of the synthetic grammar has six auxiliaries, so a
quarter of the examples use the target-only fallback, and the selection methods
coincide because the whole pool of the other rule is selected.

The Hugging Face scorer (`qcv score --backend hf`) was exercised with a stub
model and with a tiny randomly initialised LLaVA model built in code from
`transformers` classes (`tests/test_hf_scorer.py`,
`tests/test_hf_tiny_model.py`): chat-template rendering, image-token
expansion, padded batching, token-position bookkeeping and the `qcv score`
command with a locally saved model.

## Observation on the pooled residual covariance

Eq. (3) pools the residuals `r = z - c` of all calibration samples of a
signature and subtracts one global mean. A calibrated distribution is not
one-hot, so the mean residual of a block depends on the true answer (it points
from the true answer towards the rest of the simplex). Blocks whose true
answers are tied by the constraints, such as the target and a restatement or an
inverse of it, therefore have correlated residuals even when their errors are
statistically independent. In the simulator this has two visible effects when
`--correlation 0`: greedy selection in the full geometry avoids restatements
although they are independent second looks at the target, and full-metric
decoding is less accurate than the diagonal or identity metric on the same
question set (second and fourth row of the table above). The effect shrinks as
accuracy grows. When restatements share the target's errors
(`--correlation 0.8`), selection and decoding in the full geometry are ahead of
the diagonal and identity metrics. The implementation follows Eq. (3) as
written.
