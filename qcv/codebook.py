"""Feasible answer codes (Eq. 1).

A question set ``Q̄ = {q0} ∪ Q`` is encoded by concatenating one one-hot block
per question. The feasible codebook ``C_Q̄`` keeps the answer tuples that are
jointly valid under the compiler's predicates ``g_r``; ``C_Q̄(y)`` is the class
of feasible codewords whose target answer is ``y``.

Sub-pool semantics. The predicates are emitted for the whole compiled pool,
whereas selection and decoding work on subsets of it. A tuple over a subset is
feasible iff it can be extended to a jointly valid tuple over the whole pool,
i.e. the subset code is the projection of the pool-level feasible set onto the
selected questions. This is what eliminating the unselected variables from the
constraints gives (``d = b - m`` and ``t = b + m`` imply ``d = t - 2m`` on
``{d, m, t}``, as in the paper's Fig. 3) and it is the tightest code that still
contains every true answer tuple.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .constraints import evaluate_constraint
from .signatures import ordered_questions, signature_for_questions
from .types import CompiledInstance, ConstraintSpec, QuestionSpec


@dataclass(frozen=True, eq=False)
class Codebook:
    """Feasible code of one ordered question set (target first)."""

    structure_key: str
    questions: tuple[QuestionSpec, ...]
    signature: str
    answer_indices: np.ndarray
    """``(n_codewords, n_questions)`` alphabet indices in canonical product order."""
    codewords: np.ndarray
    """``(n_codewords, dimension)`` concatenated one-hot blocks."""
    target_classes: dict[str, np.ndarray]
    """Target answer ``y`` -> row indices of ``C_Q̄(y)``; empty classes are omitted."""
    block_slices: dict[str, slice]

    @property
    def dimension(self) -> int:
        return int(self.codewords.shape[1])

    @property
    def size(self) -> int:
        return int(self.codewords.shape[0])

    @property
    def target_alphabet(self) -> tuple[str, ...]:
        return self.questions[0].role.alphabet

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(len(question.role.alphabet) for question in self.questions)

    @cached_property
    def _row_of_indices(self) -> dict[tuple[int, ...], int]:
        return {tuple(int(v) for v in row): i for i, row in enumerate(self.answer_indices)}

    def assignment(self, row: int) -> dict[str, str]:
        """Answer tuple of codeword ``row`` as a ``variable -> answer`` mapping."""
        return {
            question.variable: question.role.alphabet[int(index)]
            for question, index in zip(self.questions, self.answer_indices[row], strict=True)
        }

    def _indices_of(self, assignment: Mapping[str, str]) -> tuple[int, ...]:
        indices: list[int] = []
        for question in self.questions:
            if question.variable not in assignment:
                raise KeyError(f"Missing value for {question.variable!r}")
            indices.append(question.role.answer_index(str(assignment[question.variable])))
        return tuple(indices)

    def encode_assignment(self, assignment: Mapping[str, str]) -> np.ndarray:
        """One-hot code ``c_Q̄(a)`` of an answer tuple, feasible or not."""
        return _one_hot(np.asarray([self._indices_of(assignment)]), self.block_sizes)[0]

    def row_of_indices(self, indices: Sequence[int]) -> int | None:
        """Row of the codeword with these alphabet indices; ``None`` when it is infeasible."""
        return self._row_of_indices.get(tuple(indices))

    def row_of(self, assignment: Mapping[str, str]) -> int | None:
        """Row of the codeword for ``assignment``; ``None`` when it is infeasible."""
        return self._row_of_indices.get(self._indices_of(assignment))

    def class_codewords(self, target_answer: str) -> np.ndarray:
        try:
            indices = self.target_classes[target_answer]
        except KeyError as exc:
            raise KeyError(f"Target answer {target_answer!r} has no feasible class") from exc
        return self.codewords[indices]


def _one_hot(answer_indices: np.ndarray, block_sizes: Sequence[int]) -> np.ndarray:
    rows = answer_indices.shape[0]
    offsets = np.concatenate([[0], np.cumsum(block_sizes)[:-1]]).astype(np.int64)
    codewords = np.zeros((rows, int(sum(block_sizes))), dtype=np.float64)
    if rows:
        row_index = np.repeat(np.arange(rows), len(block_sizes))
        codewords[row_index, (answer_indices + offsets[None, :]).ravel()] = 1.0
    return codewords


def enumerate_feasible_indices(
    questions: Sequence[QuestionSpec],
    constraints: Sequence[ConstraintSpec],
) -> np.ndarray:
    """All jointly valid answer tuples over ``questions`` as alphabet indices.

    Depth-first enumeration in canonical product order; each predicate is
    evaluated as soon as its last variable is assigned, so infeasible branches
    are pruned without visiting the full Cartesian product.
    """
    position = {question.variable: depth for depth, question in enumerate(questions)}
    checks_at: list[list[ConstraintSpec]] = [[] for _ in questions]
    for constraint in constraints:
        unknown = [name for name in constraint.variables if name not in position]
        if unknown:
            raise KeyError(f"Constraint {constraint.kind!r} uses unknown variables {unknown}")
        checks_at[max(position[name] for name in constraint.variables)].append(constraint)

    feasible: list[tuple[int, ...]] = []
    assignment: dict[str, str] = {}
    indices = [0] * len(questions)

    def extend(depth: int) -> None:
        if depth == len(questions):
            feasible.append(tuple(indices))
            return
        question = questions[depth]
        for index, answer in enumerate(question.role.alphabet):
            assignment[question.variable] = answer
            indices[depth] = index
            if all(evaluate_constraint(check, assignment) for check in checks_at[depth]):
                extend(depth + 1)
        del assignment[question.variable]

    extend(0)
    return np.asarray(feasible, dtype=np.int64).reshape(len(feasible), len(questions))


_POOL_CACHE: dict[str, np.ndarray] = {}
_CODEBOOK_CACHE: dict[tuple[str, tuple[str, ...]], Codebook] = {}


def pool_feasible_indices(compiled: CompiledInstance) -> np.ndarray:
    """Feasible answer tuples of the whole pool ``{q0} ∪ P(u)`` (cached per structure)."""
    cached = _POOL_CACHE.get(compiled.structure_key)
    if cached is None:
        cached = enumerate_feasible_indices(compiled.all_questions, compiled.constraints)
        cached.setflags(write=False)
        _POOL_CACHE[compiled.structure_key] = cached
    return cached


def build_codebook(
    compiled: CompiledInstance,
    questions: Sequence[QuestionSpec],
) -> Codebook:
    """Build ``C_Q̄`` and its target classes ``C_Q̄(y)`` for a subset of the pool (Eq. 1)."""
    ordered = ordered_questions(questions)
    if ordered[0].question_id != compiled.target.question_id:
        raise ValueError("The selected target does not belong to the compiled instance")
    pool = compiled.all_questions
    pool_position = {question.question_id: index for index, question in enumerate(pool)}
    question_ids = tuple(question.question_id for question in ordered)
    for question in ordered:
        position = pool_position.get(question.question_id)
        if (
            position is None
            or pool[position].variable != question.variable
            or pool[position].role.key != question.role.key
        ):
            raise ValueError("Selected questions do not all belong to the compiled instance")

    cache_key = (compiled.structure_key, question_ids)
    cached = _CODEBOOK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # The cached codebook is shared by all instances with this structure, so it
    # stores the structural question specs of the first instance that built it.
    columns = [pool_position[question_id] for question_id in question_ids]
    projected = pool_feasible_indices(compiled)[:, columns]
    # np.unique sorts rows lexicographically, which is the canonical product order.
    answer_indices = np.unique(projected, axis=0) if projected.shape[0] else projected
    block_sizes = [len(question.role.alphabet) for question in ordered]
    codewords = _one_hot(answer_indices, block_sizes)

    target_classes: dict[str, np.ndarray] = {}
    for index, answer in enumerate(ordered[0].role.alphabet):
        rows = np.flatnonzero(answer_indices[:, 0] == index)
        if rows.size:
            target_classes[answer] = rows.astype(np.int64)

    offsets = np.concatenate([[0], np.cumsum(block_sizes)]).astype(int)
    block_slices = {
        question.question_id: slice(int(offsets[i]), int(offsets[i + 1]))
        for i, question in enumerate(ordered)
    }
    for array in (answer_indices, codewords):
        array.setflags(write=False)
    codebook = Codebook(
        structure_key=f"{compiled.structure_key}:{','.join(question_ids)}",
        questions=ordered,
        signature=signature_for_questions(ordered),
        answer_indices=answer_indices,
        codewords=codewords,
        target_classes=target_classes,
        block_slices=block_slices,
    )
    _CODEBOOK_CACHE[cache_key] = codebook
    return codebook
