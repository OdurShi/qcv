import itertools

import numpy as np
import pytest
from conftest import ALL_EXAMPLES

from qcv.codebook import build_codebook, enumerate_feasible_indices, pool_feasible_indices
from qcv.constraints import evaluate_constraint


def _tuples(codebook) -> set[tuple[str, ...]]:
    return {tuple(codebook.assignment(row).values()) for row in range(codebook.size)}


def test_figure3_pool_code_is_the_set_of_scenes(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    # One codeword per (b, m) in {0, 1, 2}^2, with d = b - m and t = b + m.
    expected = {(str(b - m), str(b), str(m), str(b + m)) for b in range(3) for m in range(3)}
    assert _tuples(codebook) == expected
    assert codebook.dimension == 5 + 3 + 3 + 5
    assert np.array_equal(codebook.codewords.sum(axis=1), np.full(9, 4.0))


def test_subset_code_eliminates_unselected_variables(figure3_instance):
    compiled = figure3_instance
    target, bicycles, motorcycles, total = compiled.all_questions
    # {d, m, t}: eliminating b gives d = t - 2m (Fig. 3, third candidate set).
    codebook = build_codebook(compiled, (target, motorcycles, total))
    assert _tuples(codebook) == {
        (str(b - m), str(m), str(b + m)) for b in range(3) for m in range(3)
    }
    assert all(int(d) == int(t) - 2 * int(m) for d, m, t in _tuples(codebook))
    # {d, b, t}: eliminating m gives d = 2b - t.
    codebook = build_codebook(compiled, (target, bicycles, total))
    assert all(int(d) == 2 * int(b) - int(t) for d, b, t in _tuples(codebook))
    assert codebook.size == 9
    # {d, b}: only the existence of some m in the alphabet remains.
    codebook = build_codebook(compiled, (target, bicycles))
    assert _tuples(codebook) == {(str(b - m), str(b)) for b in range(3) for m in range(3)}
    # {d}: every difference that some scene realises.
    codebook = build_codebook(compiled, (target,))
    assert _tuples(codebook) == {(str(d),) for d in range(-2, 3)}


def test_target_classes_partition_the_code(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    rows = np.concatenate(list(codebook.target_classes.values()))
    assert sorted(rows.tolist()) == list(range(codebook.size))
    assert [len(codebook.target_classes[str(d)]) for d in range(-2, 3)] == [1, 2, 3, 2, 1]
    for answer, indices in codebook.target_classes.items():
        assert all(codebook.assignment(int(row))["d"] == answer for row in indices)
    assert codebook.class_codewords("0").shape == (3, codebook.dimension)
    with pytest.raises(KeyError):
        codebook.class_codewords("7")


def test_encoding_rows_and_canonical_order(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    scene = {"d": "1", "b": "2", "m": "1", "t": "3"}
    row = codebook.row_of(scene)
    assert row is not None
    assert np.array_equal(codebook.encode_assignment(scene), codebook.codewords[row])
    assert codebook.assignment(row) == scene
    # d = 1 but b - m = 0: a valid one-hot tuple that is not a codeword.
    assert codebook.row_of({"d": "1", "b": "2", "m": "2", "t": "4"}) is None
    ordered = [tuple(row) for row in codebook.answer_indices.tolist()]
    assert ordered == sorted(ordered)
    with pytest.raises(KeyError):
        codebook.encode_assignment({"d": "1"})
    with pytest.raises(ValueError):
        codebook.encode_assignment({"d": "9", "b": "2", "m": "1", "t": "3"})


def test_block_slices_cover_the_code_and_blocks_are_one_hot(figure3_instance):
    codebook = build_codebook(figure3_instance, figure3_instance.all_questions)
    stops = [codebook.block_slices[q.question_id] for q in codebook.questions]
    assert stops[0].start == 0 and stops[-1].stop == codebook.dimension
    for block in stops:
        assert np.array_equal(codebook.codewords[:, block].sum(axis=1), np.ones(codebook.size))


def test_build_codebook_rejects_foreign_questions(figure3_instance, compile_question):
    other = compile_question("clevr", "How many red cubes are there?")
    with pytest.raises(ValueError):
        build_codebook(figure3_instance, (figure3_instance.target, other.auxiliaries[0]))
    with pytest.raises(ValueError):
        build_codebook(figure3_instance, figure3_instance.auxiliaries)


@pytest.mark.parametrize(("task", "question"), ALL_EXAMPLES)
def test_pool_code_matches_brute_force_and_subsets_are_projections(
    compile_question, task, question
):
    compiled = compile_question(task, question)
    pool = pool_feasible_indices(compiled)
    assert pool.shape[0] > 0
    alphabets = [q.role.alphabet for q in compiled.all_questions]
    variables = [q.variable for q in compiled.all_questions]
    product_size = int(np.prod([len(alphabet) for alphabet in alphabets]))
    if product_size <= 20_000:
        brute = [
            indices
            for indices in itertools.product(*(range(len(a)) for a in alphabets))
            if all(
                evaluate_constraint(
                    constraint,
                    {
                        name: alphabets[i][index]
                        for i, (name, index) in enumerate(zip(variables, indices, strict=True))
                    },
                )
                for constraint in compiled.constraints
            )
        ]
        assert [tuple(row) for row in pool.tolist()] == brute
    # Every subset code is exactly the projection of the pool code (checked for small subsets).
    for size in (0, 1, 2):
        for subset in itertools.combinations(range(len(compiled.auxiliaries)), size):
            questions = (compiled.target, *(compiled.auxiliaries[i] for i in subset))
            codebook = build_codebook(compiled, questions)
            columns = [0, *(i + 1 for i in subset)]
            projection = {tuple(row) for row in pool[:, columns].tolist()}
            assert {tuple(row) for row in codebook.answer_indices.tolist()} == projection
            assert len(codebook.target_classes) >= 1


def test_enumeration_prunes_without_changing_the_result(figure3_instance):
    questions = figure3_instance.all_questions
    forward = enumerate_feasible_indices(questions, figure3_instance.constraints)
    assert forward.shape == (9, 4)
    with pytest.raises(KeyError):
        enumerate_feasible_indices(questions[:2], figure3_instance.constraints)


def test_one_hot_relations_constrain_small_subsets(compile_question):
    """A single Boolean probe must already be tied to the relation it tests."""
    compiled = compile_question(
        "clevr", "What is the spatial relation of cube a relative to cube b?"
    )
    is_left = next(q for q in compiled.auxiliaries if q.variable == "is_left")
    codebook = build_codebook(compiled, (compiled.target, is_left))
    assert _tuples(codebook) == {
        ("left", "yes"),
        ("right", "no"),
        ("front", "no"),
        ("behind", "no"),
    }
