"""Synthetic stand-in for a dataset and a frozen VLM.

The simulator makes the whole pipeline runnable on a laptop without images or
model weights. It is *not* a model of any benchmark in the paper, and numbers
obtained with it say nothing about real VLMs.

Scene model. A raw record is a question of the ``synthetic`` task grammar with
random entity names. Its latent scene state is one jointly valid answer tuple
of the compiled pool, drawn uniformly from the pool-level feasible code; the
tuple is stored as the record's ``gold`` annotation and plays the role that the
image plays for a real VLM.

Score model. Every question ``k`` is answered from a *percept* ``a_k``: the true
answer with probability ``1 - error_rate`` and a uniformly random wrong answer
otherwise. The simulated length-normalised log-scores are

    l_k = log_softmax( signal * e(a_k) + noise * eps_k ),   eps_k ~ N(0, I),

so the simulated model is confident in its percept, right or wrong. Percepts
are independent across questions except for auxiliaries that merely restate the
target (roles in ``restatement_roles``; same answer alphabet as the target):
with probability ``correlation`` a restatement copies the target's percept,
mistake included, and otherwise draws its own. The error indicators of the
target and of a restatement therefore have correlation ``correlation``, while
the marginal error rate of every single question is ``error_rate`` for every
value of ``correlation``. With ``correlation = 0`` restatements are independent
second looks at the target; close to one they are redundant, and only the
inverse and part questions carry independent evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .codebook import pool_feasible_indices
from .compilers import CompilerRegistry, GrammarCompiler
from .types import CompiledInstance, QuestionSpec
from .utils import stable_hash

SYNTHETIC_TASK = "synthetic"
_COLOURS = ("red", "blue", "green", "yellow", "purple", "gray", "brown", "cyan")
_SHAPES = ("cube", "sphere", "cylinder", "cone")
_COUNT_TEMPLATE = "How many more {a} than {b} are there?"
_RELATION_TEMPLATE = "What is the spatial relation of {a} relative to {b}?"


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the simulator."""

    error_rate: float = 0.1
    """Probability that the percept of a question is a wrong answer."""
    correlation: float = 0.8
    """Probability that a restatement of the target copies the target's percept."""
    signal: float = 4.0
    """Log-score advantage of the perceived answer."""
    noise: float = 1.0
    """Standard deviation of the independent log-score jitter of every alphabet entry."""
    relation_fraction: float = 0.25
    """Fraction of records drawn from the inverse-relation rule."""
    restatement_roles: tuple[str, ...] = (
        "difference_restated",
        "difference_recomputed",
        "spatial_relation_restated",
    )
    """Roles whose questions restate the target and may inherit its percept."""
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("error_rate", "correlation", "relation_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.noise < 0.0:
            raise ValueError("noise must be non-negative")


def _generator(*parts: object) -> np.random.Generator:
    return np.random.default_rng(int(stable_hash([str(part) for part in parts], length=16), 16))


def _synthetic_compiler() -> GrammarCompiler:
    return CompilerRegistry.from_directory().compiler_for(SYNTHETIC_TASK)


def sample_scene_state(
    compiled: CompiledInstance, generator: np.random.Generator
) -> dict[str, str]:
    """Draw a jointly valid answer tuple of the whole pool, uniformly over the feasible code."""
    feasible = pool_feasible_indices(compiled)
    row = feasible[int(generator.integers(0, feasible.shape[0]))]
    return {
        question.variable: question.role.alphabet[int(index)]
        for question, index in zip(compiled.all_questions, row, strict=True)
    }


def generate_raw_records(count: int, split: str, config: SyntheticConfig) -> list[dict[str, Any]]:
    """Generate raw records (question, answer, gold scene state) for one split."""
    compiler = _synthetic_compiler()
    generator = _generator(config.seed, "records", split)
    records: list[dict[str, Any]] = []
    entities = [f"{colour} {shape}" for colour in _COLOURS for shape in _SHAPES]
    for index in range(count):
        first, second = (
            entities[int(i)] for i in generator.choice(len(entities), 2, replace=False)
        )
        if generator.random() < config.relation_fraction:
            question = _RELATION_TEMPLATE.format(a=f"the {first}", b=f"the {second}")
        else:
            question = _COUNT_TEMPLATE.format(a=f"{first}s", b=f"{second}s")
        instance_id = f"synthetic-{split}-{index:06d}"
        result = compiler.compile(instance_id, question)
        if result.compiled is None:
            raise RuntimeError(f"Synthetic question failed to compile: {result.reason}")
        gold = sample_scene_state(result.compiled, generator)
        records.append(
            {
                "instance_id": instance_id,
                "task": SYNTHETIC_TASK,
                "question": question,
                "answer": gold[result.compiled.target.variable],
                "gold": gold,
                "media": {"type": "none"},
                "metadata": {"split": split, "rule": result.compiled.rule_name},
            }
        )
    return records


class SyntheticScorer:
    """Simulated frozen VLM: returns full-alphabet log-scores for a question."""

    def __init__(self, config: SyntheticConfig):
        self.config = config

    def _own_percept(self, question: QuestionSpec, gold_index: int, instance_id: str) -> int:
        """Independent percept of one question: the truth, or a uniform wrong answer."""
        generator = _generator(self.config.seed, "percept", instance_id, question.question_id)
        size = len(question.role.alphabet)
        if size < 2 or generator.random() >= self.config.error_rate:
            return gold_index
        return int((gold_index + 1 + generator.integers(0, size - 1)) % size)

    def percept(
        self, question: QuestionSpec, target: QuestionSpec, gold: dict[str, str], instance_id: str
    ) -> int:
        """Alphabet index of the answer the simulated model perceives for ``question``."""
        gold_index = question.role.answer_index(str(gold[question.variable]))
        restates_target = (
            not question.is_target
            and question.role.name in self.config.restatement_roles
            and question.role.alphabet == target.role.alphabet
        )
        if restates_target:
            generator = _generator(self.config.seed, "copy", instance_id, question.question_id)
            if generator.random() < self.config.correlation:
                target_gold = target.role.answer_index(str(gold[target.variable]))
                return self._own_percept(target, target_gold, instance_id)
        return self._own_percept(question, gold_index, instance_id)

    def score_question(
        self, question: QuestionSpec, target: QuestionSpec, gold: dict[str, str], instance_id: str
    ) -> tuple[float, ...]:
        """Simulated ``l_k(a)`` for every answer ``a`` of the question's alphabet."""
        config = self.config
        generator = _generator(config.seed, "jitter", instance_id, question.question_id)
        logits = config.noise * generator.standard_normal(len(question.role.alphabet))
        logits[self.percept(question, target, gold, instance_id)] += config.signal
        logits = logits - np.max(logits)
        log_scores = logits - np.log(np.sum(np.exp(logits)))
        return tuple(float(value) for value in log_scores)
