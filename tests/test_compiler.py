import pytest
from conftest import ALL_EXAMPLES

from qcv.compilers import GrammarCompiler, default_grammar_directory, load_task_grammar
from qcv.types import CompiledInstance


def test_packaged_grammars_are_the_single_source_of_truth(registry):
    names = sorted(path.stem for path in default_grammar_directory().glob("*.yaml"))
    assert names == ["agqa", "chartqa", "clevr", "gqa", "synthetic"]
    assert registry.tasks == ("agqa", "chartqa", "clevr", "gqa", "synthetic")


@pytest.mark.parametrize(("task", "question"), ALL_EXAMPLES)
def test_every_grammar_rule_compiles(compile_question, task, question):
    compiled = compile_question(task, question)
    assert compiled.target.is_target and compiled.target.question_id == "q0"
    indices = [question.template_index for question in compiled.auxiliaries]
    assert indices == sorted(indices) and len(set(indices)) == len(indices)
    variables = [question.variable for question in compiled.all_questions]
    assert len(set(variables)) == len(variables)
    for constraint in compiled.constraints:
        assert set(constraint.variables) <= set(variables)
    assert CompiledInstance.from_dict(compiled.to_dict()) == compiled


def test_count_compiler_is_deterministic_and_text_normalised(compile_question):
    first = compile_question("clevr", "How many red cubes are there?")
    second = compile_question("clevr", "How many   RED cubes are there ?")
    assert first.rule_name == "count_complement"
    assert first.target.role.name == "count"
    assert [q.variable for q in first.auxiliaries] == [
        "exists_a",
        "count_not_a",
        "count_total",
        "exists_not_a",
    ]
    assert [q.template_index for q in first.auxiliaries] == [1, 2, 3, 4]
    assert first.constraints == second.constraints
    assert first.structure_key == second.structure_key


def test_structure_key_ignores_entities_but_not_rules(compile_question):
    cubes = compile_question("clevr", "How many red cubes are there?", "a")
    balls = compile_question("clevr", "How many blue balls are there?", "b")
    existence = compile_question("clevr", "Are there any red cubes?", "c")
    assert cubes.structure_key == balls.structure_key
    assert cubes.structure_key != existence.structure_key


def test_unmatched_question_is_rejected(registry):
    result = registry.compiler_for("clevr").compile("x", "What color is the cube?")
    assert result.compiled is None
    assert result.reason == "no_unique_schema_match"


def test_ambiguous_match_is_rejected(tmp_path):
    grammar = tmp_path / "toy.yaml"
    grammar.write_text(
        """
task: toy
roles:
  flag: {answer_type: boolean, values: ["yes", "no"]}
rules:
  - name: first
    pattern: "is (?P<a>.+)"
    target: {role: flag, variable: x}
  - name: second
    pattern: "is it (?P<a>.+)"
    target: {role: flag, variable: x}
""",
        encoding="utf-8",
    )
    result = GrammarCompiler(load_task_grammar(grammar)).compile("x", "Is it red?")
    assert result.compiled is None
    assert result.reason == "ambiguous_schema_match"
    assert result.matching_rules == ("first", "second")


def test_duplicate_auxiliary_text_is_dropped_and_infeasible_pool_rejected(tmp_path):
    grammar = tmp_path / "toy.yaml"
    grammar.write_text(
        """
task: toy
roles:
  flag: {answer_type: boolean, values: ["yes", "no"]}
  other: {answer_type: boolean, values: ["yes", "no"]}
rules:
  - name: dedupe
    pattern: "is (?P<a>.+) red"
    target: {role: flag, variable: x}
    auxiliaries:
      - {index: 2, text: "Is {a} not red?", role: other, variable: y}
      - {index: 1, text: "is {a}  NOT red", role: other, variable: z}
  - name: impossible
    pattern: "is (?P<a>.+) blue"
    target: {role: flag, variable: x}
    auxiliaries:
      - {index: 1, text: "Is {a} not blue?", role: other, variable: y}
    constraints:
      - {kind: equal, variables: [x, y]}
      - {kind: boolean_not, variables: [x, y]}
""",
        encoding="utf-8",
    )
    compiler = GrammarCompiler(load_task_grammar(grammar))
    deduplicated = compiler.compile("x", "Is the cube red?").compiled
    assert deduplicated is not None
    assert [q.variable for q in deduplicated.auxiliaries] == ["z"]
    assert compiler.compile("x", "Is the cube blue?").reason == "empty_feasible_code"


def test_decimal_range_alphabet(tmp_path):
    grammar = tmp_path / "toy.yaml"
    grammar.write_text(
        """
task: toy
roles:
  value:
    answer_type: numeric
    decimal_range: {start: 0, stop: 1, step: 0.25, precision: 2}
rules: []
""",
        encoding="utf-8",
    )
    assert load_task_grammar(grammar).roles["value"].alphabet == ("0", "0.25", "0.5", "0.75", "1")


@pytest.mark.parametrize(
    "question",
    [
        "Are there an equal number of red cubes and blue spheres?",
        "Is the number of cubes left of the ball greater than the number of spheres?",
        "Is the number of spheres that are left of the cube the same as the number of things?",
        "Is the shape of the big thing the same as the small thing that is left of the cube?",
        "Is the tiny sphere made of the same material as the large thing left of the cube?",
    ],
)
def test_clevr_comparison_questions_are_not_compiled_as_existence_or_relation(registry, question):
    result = registry.compiler_for("clevr").compile("x", question)
    assert result.compiled is None and result.reason == "no_unique_schema_match"


def test_clevr_existence_questions_with_same_attribute_phrases_still_compile(compile_question):
    question = "Is there a big brown object of the same shape as the green thing?"
    assert compile_question("clevr", question).rule_name == "existence_count"
