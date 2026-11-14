"""Data directory resolution, shipped data, download tooling and dataset converters."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from conftest import REPOSITORY_ROOT, load_script
from PIL import Image

from qcv.data import DATA_DIR_VARIABLE, default_data_dir
from qcv.data.download import download_dataset, ensure_file, sha256_of
from qcv.data.sources import SOURCES, DatasetSource, RemoteFile
from qcv.pipeline import provide_synthetic_raw_jsonl, synthesize_raw_jsonl
from qcv.synthetic import SyntheticConfig
from qcv.utils import read_jsonl, write_jsonl

SHIPPED = REPOSITORY_ROOT / "data" / "synthetic"


def test_data_dir_resolution_order(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    assert default_data_dir("explicit") == Path("explicit")
    # No ./data here: a source checkout (editable install) falls back to its own data directory.
    import qcv

    from_checkout = Path(qcv.__file__).resolve().parents[1] == REPOSITORY_ROOT
    expected = REPOSITORY_ROOT / "data" if from_checkout else Path("data")
    assert default_data_dir() == expected
    (tmp_path / "data").mkdir()
    # ./data is returned as a relative path so that record paths stay portable.
    assert default_data_dir() == Path("data")
    monkeypatch.setenv(DATA_DIR_VARIABLE, str(tmp_path / "elsewhere"))
    assert default_data_dir() == tmp_path / "elsewhere"
    assert default_data_dir("explicit") == Path("explicit")


def test_shipped_synthetic_records_match_their_manifest_and_the_generator(tmp_path):
    manifest = json.loads((SHIPPED / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        assert sha256_of(SHIPPED / name) == entry["sha256"]
        assert sum(1 for _ in read_jsonl(SHIPPED / name)) == entry["records"]
    config = SyntheticConfig(seed=manifest["seed"], relation_fraction=manifest["relation_fraction"])
    synthesize_raw_jsonl(tmp_path, manifest["calibration_size"], manifest["test_size"], config)
    for name in manifest["files"]:
        assert (tmp_path / name).read_bytes() == (SHIPPED / name).read_bytes()
    regenerated = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert regenerated == manifest


def test_demo_records_come_from_data_when_the_manifest_matches(tmp_path):
    data_dir = tmp_path / "data"
    synthesize_raw_jsonl(data_dir / "synthetic", 30, 40, SyntheticConfig(seed=7))
    marker = {"instance_id": "marker", "task": "synthetic", "question": "q"}
    write_jsonl(data_dir / "synthetic" / "test_raw.jsonl", [marker])

    used = provide_synthetic_raw_jsonl(tmp_path / "a", 30, 40, SyntheticConfig(seed=7), data_dir)
    assert used == "data"
    assert list(read_jsonl(tmp_path / "a" / "test_raw.jsonl")) == [marker]
    # A different seed or size is not what was shipped, so the records are generated.
    used = provide_synthetic_raw_jsonl(tmp_path / "b", 30, 40, SyntheticConfig(seed=8), data_dir)
    assert used == "generated"
    assert sum(1 for _ in read_jsonl(tmp_path / "b" / "test_raw.jsonl")) == 40
    used = provide_synthetic_raw_jsonl(tmp_path / "c", 30, 41, SyntheticConfig(seed=7), data_dir)
    assert used == "generated"
    used = provide_synthetic_raw_jsonl(
        tmp_path / "d", 30, 40, SyntheticConfig(seed=7), tmp_path / "missing"
    )
    assert used == "generated"


def _archive(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in members.items():
            bundle.writestr(name, content)
    return path


def _local_source(tmp_path: Path) -> DatasetSource:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    annotations = _archive(upstream / "annotations.zip", {"toy/questions.json": '{"q": 1}'})
    media = _archive(upstream / "images.zip", {"toy/images/a.txt": "pixels"})
    return DatasetSource(
        key="toy",
        title="Toy",
        homepage="https://example.org/toy",
        licence="CC0",
        version="1",
        files=(
            RemoteFile(annotations.as_uri(), annotations.stat().st_size),
            RemoteFile(media.as_uri(), media.stat().st_size, media=True),
        ),
    )


def test_download_unpacks_records_checksums_and_skips_known_files(tmp_path):
    source = _local_source(tmp_path)
    data_dir = tmp_path / "data"
    manifest = download_dataset(source, data_dir)
    assert (data_dir / "toy" / "toy" / "questions.json").read_text() == '{"q": 1}'
    assert not (data_dir / "toy" / "toy" / "images").exists()  # media only on request
    entry = manifest["files"]["annotations.zip"]
    archive = data_dir / "_downloads" / "annotations.zip"
    assert entry["sha256"] == sha256_of(archive) and entry["bytes"] == archive.stat().st_size
    assert manifest["licence"] == "CC0" and manifest["homepage"] == "https://example.org/toy"
    stored = json.loads((data_dir / "toy" / "DOWNLOAD.json").read_text(encoding="utf-8"))
    assert stored == manifest

    # A recorded file is not fetched again: the upstream copy can disappear.
    (tmp_path / "upstream" / "annotations.zip").unlink()
    assert download_dataset(source, data_dir)["files"] == manifest["files"]
    with pytest.raises(OSError):
        download_dataset(source, data_dir, force=True)

    with_media = download_dataset(source, data_dir, include_media=True)
    assert set(with_media["files"]) == {"annotations.zip", "images.zip"}
    assert (data_dir / "toy" / "toy" / "images" / "a.txt").read_text() == "pixels"


def test_ensure_file_downloads_only_when_the_file_is_absent(tmp_path, monkeypatch):
    source = _local_source(tmp_path)
    monkeypatch.setitem(SOURCES, "toy", source)
    data_dir = tmp_path / "data"
    wanted = data_dir / "toy" / "toy" / "questions.json"
    assert ensure_file(wanted, "toy", data_dir) == wanted and wanted.is_file()
    (tmp_path / "upstream" / "annotations.zip").unlink()
    assert ensure_file(wanted, "toy", data_dir) == wanted  # present: no network access
    with pytest.raises(FileNotFoundError):
        ensure_file(data_dir / "toy" / "toy" / "missing.json", "toy", data_dir)


def test_source_registry_covers_the_benchmark_grammars(registry):
    assert set(SOURCES) == set(registry.tasks) - {"synthetic"}
    for source in SOURCES.values():
        assert source.homepage.startswith("https://") and source.licence
        assert any(not remote.media for remote in source.files)
        for remote in source.files:
            assert remote.url.startswith("https://") and remote.size_bytes > 0
            assert remote.name.endswith(".zip")


def _run_script(name: str, *arguments, cwd: Path = REPOSITORY_ROOT) -> dict:
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY_ROOT / "scripts" / name), *(str(a) for a in arguments)],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_clevr_converter_reads_the_default_data_layout(tmp_path):
    root = tmp_path / "clevr" / "CLEVR_v1.0"
    (root / "questions").mkdir(parents=True)
    (root / "scenes").mkdir()
    objects = [{"color": "red", "shape": "cube"}, {"color": "blue", "shape": "sphere"}]
    scenes = {"scenes": [{"image_filename": "CLEVR_val_000000.png", "objects": objects}]}
    count_program = [
        {"function": "scene", "inputs": [], "value_inputs": []},
        {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
        {"function": "count", "inputs": [1], "value_inputs": []},
    ]
    questions = {
        "info": {"split": "val"},
        "questions": [
            {"question_index": 0, "image_filename": "CLEVR_val_000000.png",
             "question": "How many red things are there?", "answer": "1",
             "program": count_program},
            {"question_index": 1, "image_filename": "CLEVR_val_000000.png",
             "question": "What color is the cube?", "answer": "red", "program": []},
        ],
    }  # fmt: skip
    (root / "questions" / "CLEVR_val_questions.json").write_text(json.dumps(questions))
    (root / "scenes" / "CLEVR_val_scenes.json").write_text(json.dumps(scenes))

    summary = _run_script("convert_clevr.py", "--data-dir", tmp_path, "--drop-incomplete-gold")
    output = tmp_path / "clevr" / "clevr_val_raw.jsonl"
    assert summary == {"questions": 2, "compiled": 1, "written": 1, "output": str(output)}
    (row,) = read_jsonl(output)
    assert row["instance_id"] == "clevr-val-000000" and row["answer"] == "1"
    assert row["media"]["path"] == str(root / "images" / "val" / "CLEVR_val_000000.png")
    assert row["gold"] == {
        "count_a": "1",
        "exists_a": "yes",
        "count_not_a": "1",
        "count_total": "2",
        "exists_not_a": "yes",
    }


def test_generic_converter_handles_id_maps_and_lists_inside_the_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "gqa").mkdir(parents=True)
    questions = {
        "201": {"question": "Are there any cars?", "answer": "yes", "imageId": "n1"},
        "202": {"question": "How many cars are there?", "answer": "2", "imageId": "n2"},
    }
    (data_dir / "gqa" / "val_balanced_questions.json").write_text(json.dumps(questions))
    (data_dir / "gqa" / "gold.json").write_text(json.dumps({"201": {"count_a": 2}}))
    summary = _run_script(
        "convert_records.py", "--data-dir", data_dir, "--task", "gqa",
        "--input", "gqa/val_balanced_questions.json", "--question-field", "question",
        "--answer-field", "answer", "--media-field", "imageId", "--media-suffix", ".jpg",
        "--media-root", "gqa/images", "--gold-sidecar", "gqa/gold.json", cwd=tmp_path,
    )  # fmt: skip
    assert summary == {"written": 2, "output": str(data_dir / "gqa" / "gqa_raw.jsonl")}
    first, second = read_jsonl(data_dir / "gqa" / "gqa_raw.jsonl")
    assert first["instance_id"] == "201" and second["instance_id"] == "202"
    assert first["media"] == {"type": "image", "path": str(data_dir / "gqa/images" / "n1.jpg")}
    assert first["gold"] == {"count_a": "2"} and second["gold"] == {}

    convert_records = load_script("convert_records")
    rows = convert_records.convert(
        [{"imgname": "a.png", "query": "Is A greater than B?", "label": "Yes"}],
        task="chartqa", question_field="query", answer_field="label", media_field="imgname",
    )  # fmt: skip
    assert rows[0]["instance_id"] == "chartqa-0000000" and rows[0]["media"]["path"] == "a.png"
    with pytest.raises(ValueError):
        (tmp_path / "bad.json").write_text("3")
        list(convert_records.load_records(tmp_path / "bad.json"))


def test_split_is_deterministic_complete_and_order_independent(tmp_path):
    rows = [{"instance_id": f"example-{index:04d}", "value": index} for index in range(2000)]
    write_jsonl(tmp_path / "raw.jsonl", rows)
    summary = _run_script("split_records.py", "--input", tmp_path / "raw.jsonl",
                          "--calibration-fraction", "0.2", "--seed", "3")  # fmt: skip
    calibration = list(read_jsonl(tmp_path / "raw.calibration.jsonl"))
    test = list(read_jsonl(tmp_path / "raw.test.jsonl"))
    assert summary["calibration_records"] == len(calibration)
    assert len(calibration) + len(test) == 2000
    assert len(calibration) == pytest.approx(400, abs=60)
    ids = {row["instance_id"] for row in calibration}
    assert not ids & {row["instance_id"] for row in test}

    write_jsonl(tmp_path / "reversed.jsonl", rows[::-1])
    _run_script("split_records.py", "--input", tmp_path / "reversed.jsonl", "--seed", "3")
    again = {row["instance_id"] for row in read_jsonl(tmp_path / "reversed.calibration.jsonl")}
    assert again == ids
    outputs = (
        "--calibration-output",
        tmp_path / "c4.jsonl",
        "--test-output",
        tmp_path / "t4.jsonl",
    )
    _run_script("split_records.py", "--input", tmp_path / "raw.jsonl", "--seed", "4", *outputs)
    assert {row["instance_id"] for row in read_jsonl(tmp_path / "c4.jsonl")} != ids


def test_paper_figures_are_present_and_documented():
    extract = load_script("extract_paper_figures")
    folder = REPOSITORY_ROOT / "assets" / "paper_figures"
    listing = (folder / "README.md").read_text(encoding="utf-8")
    assert list(extract.FIGURES) == ["fig01", "fig02", "fig03", "fig04", "fig05"]
    for name, (page, (x0, y0, x1, y1)) in extract.FIGURES.items():
        assert 1 <= page <= 8 and 0 <= x0 < x1 <= 612 and 0 <= y0 < y1 <= 792
        image_path = folder / f"{name}.png"
        assert image_path.stat().st_size < 1_500_000
        with Image.open(image_path) as image:
            # 200 dpi rendering of the crop box (72 points per inch).
            assert image.size[0] == pytest.approx((x1 - x0) * 200 / 72, abs=2)
            assert image.size[1] == pytest.approx((y1 - y0) * 200 / 72, abs=2)
        assert f"`{name}.png`" in listing


CLEVR = REPOSITORY_ROOT / "data" / "clevr"


@pytest.fixture(scope="module")
def clevr_manifest() -> dict:
    return json.loads((CLEVR / "MANIFEST.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clevr_records(clevr_manifest) -> dict[str, list[dict]]:
    return {name: list(read_jsonl(CLEVR / name)) for name in clevr_manifest["files"]}


def test_shipped_clevr_records_match_their_manifest(clevr_manifest, clevr_records):
    import gzip
    import hashlib

    assert set(clevr_manifest["files"]) == {
        "clevr_val_calibration.jsonl.gz",
        "clevr_val_test.jsonl.gz",
    }
    for name, entry in clevr_manifest["files"].items():
        path = CLEVR / name
        assert path.stat().st_size == entry["bytes"] < 50_000_000
        assert sha256_of(path) == entry["sha256"]
        with gzip.open(path, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == entry["uncompressed_sha256"]
        assert len(clevr_records[name]) == entry["records"] == sum(entry["rules"].values())
    selection = clevr_manifest["selection"]
    assert sum(len(rows) for rows in clevr_records.values()) == selection["records"]
    assert selection["records"] == selection["compiled"] <= selection["questions"] == 149_991
    # The records were compiled with the grammar that is shipped now.
    grammar = REPOSITORY_ROOT / clevr_manifest["grammar"]["file"]
    assert sha256_of(grammar) == clevr_manifest["grammar"]["sha256"]
    # Licence and attribution travel with the data.
    assert clevr_manifest["licence"] == "CC BY 4.0"
    assert "Attribution 4.0 International" in (CLEVR / "LICENSE.txt").read_text(encoding="utf-8")
    attribution = (CLEVR / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in attribution and "Changes made" in attribution


def test_shipped_clevr_partition_is_the_documented_hash_split(clevr_manifest, clevr_records):
    in_calibration = load_script("split_records").in_calibration
    partition = clevr_manifest["partition"]
    seen: set[str] = set()
    for name, rows in clevr_records.items():
        expected = "calibration" in name
        for row in rows:
            instance_id = row["instance_id"]
            assert instance_id not in seen
            seen.add(instance_id)
            chosen = in_calibration(
                instance_id, partition["calibration_fraction"], partition["seed"]
            )
            assert chosen is expected
    assert len(seen) == clevr_manifest["selection"]["records"]


def test_shipped_clevr_records_compile_and_carry_feasible_gold(registry, clevr_records):
    from qcv.codebook import build_codebook
    from qcv.data import validate_raw_record

    compiler = registry.compiler_for("clevr")
    rules: dict[str, int] = {}
    for rows in clevr_records.values():
        for row in rows:
            assert validate_raw_record(row)["task"] == "clevr"
            media = row["media"]["path"]
            # Portable image paths: relative to the repository, in the download layout.
            assert media == f"data/clevr/CLEVR_v1.0/images/val/{row['metadata']['image_filename']}"
            compiled = compiler.compile(row["instance_id"], row["question"]).compiled
            assert compiled is not None
            rules[compiled.rule_name] = rules.get(compiled.rule_name, 0) + 1
            assert row["gold"][compiled.target.variable] == row["answer"]
            assert set(row["gold"]) == {question.variable for question in compiled.all_questions}
            assert build_codebook(compiled, compiled.all_questions).row_of(row["gold"]) is not None
    assert set(rules) == {"count_complement", "existence_count"}


def test_pipeline_runs_on_the_shipped_clevr_records(clevr_records, tmp_path):
    """compile -> score (simulated VLM) -> calibrate -> infer -> evaluate on CLEVR records."""
    from qcv.inference import InferenceOptions
    from qcv.pipeline import (
        calibrate_jsonl,
        compile_jsonl,
        evaluate_jsonl,
        infer_jsonl,
        score_jsonl_synthetic,
    )

    config = SyntheticConfig(seed=0)
    sizes = {"clevr_val_calibration.jsonl.gz": 600, "clevr_val_test.jsonl.gz": 300}
    scores = {}
    for name, size in sizes.items():
        stem = name.split(".")[0]
        write_jsonl(tmp_path / f"{stem}.jsonl.gz", clevr_records[name][:size])
        summary = compile_jsonl(tmp_path / f"{stem}.jsonl.gz", tmp_path / f"{stem}.compiled.jsonl")
        assert summary == {"compiled": size, "rejected": 0, "written": size}
        scores[name] = tmp_path / f"{stem}.scores.jsonl"
        score_jsonl_synthetic(tmp_path / f"{stem}.compiled.jsonl", scores[name], config)
    calibrate_jsonl(scores["clevr_val_calibration.jsonl.gz"], tmp_path / "artifacts", max_budget=3)
    test_scores = scores["clevr_val_test.jsonl.gz"]
    counts = infer_jsonl(
        test_scores, tmp_path / "artifacts", tmp_path / "qcv.jsonl", InferenceOptions(budget=3)
    )
    assert counts == {"predictions": 300, "covered": 300, "uncovered": 0}
    metrics = evaluate_jsonl(
        test_scores, tmp_path / "qcv.jsonl", tmp_path / "qcv.json", tmp_path / "artifacts"
    )
    assert metrics["total"] == 300 and set(metrics["per_task"]) == {"clevr"}
    assert metrics["separation_audit"]["sufficient_condition_violations"] == 0


def test_build_clevr_slice_script_on_a_small_clevr_tree(tmp_path):
    root = tmp_path / "clevr" / "CLEVR_v1.0"
    (root / "questions").mkdir(parents=True)
    (root / "scenes").mkdir()
    (root / "COPYRIGHT.txt").write_text("CLEVR (c) 2017, Example.\n\nLicensed under CC BY 4.0.\n")
    (root / "LICENSE.txt").write_text("Attribution 4.0 International\n")
    objects = [{"color": "red", "shape": "cube"}, {"color": "blue", "shape": "sphere"}]
    scenes = {"scenes": [{"image_filename": "CLEVR_val_000000.png", "objects": objects}]}
    scene_node = {"function": "scene", "inputs": [], "value_inputs": []}
    questions = []
    for index in range(60):
        colour = ("red", "blue", "green")[index % 3]
        program = [
            scene_node,
            {"function": "filter_color", "inputs": [0], "value_inputs": [colour]},
            {"function": "count", "inputs": [1], "value_inputs": []},
        ]
        questions.append(
            {"question_index": index, "image_filename": "CLEVR_val_000000.png",
             "question": f"How many {colour} things are there?",
             "answer": "0" if colour == "green" else "1", "program": program}
        )  # fmt: skip
    questions.append(
        {"question_index": 60, "image_filename": "CLEVR_val_000000.png",
         "question": "What color is the cube?", "answer": "red", "program": []}
    )  # fmt: skip
    (root / "questions" / "CLEVR_val_questions.json").write_text(
        json.dumps({"info": {"split": "val"}, "questions": questions})
    )
    (root / "scenes" / "CLEVR_val_scenes.json").write_text(json.dumps(scenes))

    manifest = _run_script("build_clevr_slice.py", "--data-dir", tmp_path, "--seed", "1")
    target = tmp_path / "clevr"
    assert manifest == json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["selection"] == {"split": "val", "questions": 61, "compiled": 60, "records": 60}
    assert manifest["partition"]["seed"] == 1 and "archive" in manifest["source"]
    total = 0
    for name, entry in manifest["files"].items():
        rows = list(read_jsonl(target / name))
        assert len(rows) == entry["records"] and sha256_of(target / name) == entry["sha256"]
        assert entry["rules"] == ({"count_complement": len(rows)} if rows else {})
        total += len(rows)
    assert total == 60
    assert (target / "LICENSE.txt").read_text() == "Attribution 4.0 International\n"
    attribution = (target / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert "CLEVR (c) 2017, Example." in attribution and "(60 of 61)" in attribution


def test_fetch_falls_back_to_curl_when_python_has_no_ca_bundle(tmp_path, monkeypatch):
    import shutil
    import ssl
    import urllib.error

    from qcv.data import download

    if shutil.which("curl") is None:
        pytest.skip("curl is not installed")
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")

    class Unverifiable:
        def open(self, url):
            raise urllib.error.URLError(ssl.SSLCertVerificationError("no CA bundle"))

    monkeypatch.setattr(download.urllib.request, "build_opener", lambda *handlers: Unverifiable())
    fetched = download.fetch(source.as_uri(), tmp_path / "out" / "copy.bin")
    assert fetched.read_bytes() == b"payload"
    assert not (tmp_path / "out" / "copy.bin.part").exists()

    class Offline:
        def open(self, url):
            raise urllib.error.URLError(OSError("network unreachable"))

    monkeypatch.setattr(download.urllib.request, "build_opener", lambda *handlers: Offline())
    with pytest.raises(urllib.error.URLError):
        download.fetch(source.as_uri(), tmp_path / "out" / "other.bin")
