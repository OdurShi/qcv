"""Ordered role signatures: the unit over which residual covariance is estimated."""

from __future__ import annotations

from collections.abc import Sequence

from .types import QuestionSpec
from .utils import stable_hash


def ordered_questions(questions: Sequence[QuestionSpec]) -> tuple[QuestionSpec, ...]:
    """Canonical order of a question set: target first, auxiliaries by template index."""
    targets = [question for question in questions if question.is_target]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target question, found {len(targets)}")
    auxiliaries = sorted(
        (question for question in questions if not question.is_target),
        key=lambda item: (item.template_index, item.text, item.question_id),
    )
    return (targets[0], *auxiliaries)


def role_keys(questions: Sequence[QuestionSpec]) -> tuple[str, ...]:
    return tuple(question.role.key for question in ordered_questions(questions))


def signature_for_questions(questions: Sequence[QuestionSpec]) -> str:
    """Hash of the ordered (role, alphabet) keys; samples are pooled only within one signature."""
    ordered = ordered_questions(questions)
    keys = [question.role.key for question in ordered]
    return stable_hash(keys, length=32)
