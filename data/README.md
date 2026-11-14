# Data

All commands resolve this directory by default (`--data-dir`, the `QCV_DATA_DIR` environment variable, or `./data`). Files that are present are used as they are; the benchmark archives are fetched by `scripts/download_data.py`.

```text
data/
  README.md
  synthetic/            shipped: synthetic scenes for the quick start, the tests and results/
    calibration_raw.jsonl
    test_raw.jsonl
    MANIFEST.json
  clevr/                shipped: CLEVR validation questions compiled by the CLEVR grammar
    clevr_val_calibration.jsonl.gz
    clevr_val_test.jsonl.gz
    MANIFEST.json  LICENSE.txt  ATTRIBUTION.md
    CLEVR_v1.0/         fetched: python scripts/download_data.py clevr [--media]
  gqa/                  fetched: python scripts/download_data.py gqa
  chartqa/              fetched: python scripts/download_data.py chartqa
  agqa/                 fetched: python scripts/download_data.py agqa
  _downloads/           the fetched archives
```

Image and video archives are fetched only with `--media`. Every fetched archive is recorded with its URL, size and SHA-256 in `data/<dataset>/DOWNLOAD.json`. Git tracks `README.md`, `synthetic/` and the five files of `clevr/` listed above; archives, their unpacked contents and intermediate files are ignored.

## Benchmarks used in the paper

The paper evaluates on compiler-supported slices of four benchmarks with a held-out calibration split (Table 1, Table 3, Figures 2, 4 and 5). The benchmarks are distributed by their maintainers: their annotation archives range from 89 MB to 1.5 GB and the media archives from 16 to 22 GB. CLEVR is released under CC BY 4.0 and ChartQA under GPL-3.0; the GQA and AGQA-Decomp pages state no licence. A compact set of CLEVR records is shipped (next section); everything else is fetched. Sizes are the values reported by the servers on 2026-11-13.

| | CLEVR | GQA | ChartQA | AGQA-Decomp |
|---|---|---|---|---|
| Content | synthetic 3D scenes; counting, existence and spatial questions with scene graphs and functional programs | real images; compositional questions with scene graphs | charts; questions over values with data tables | videos; compositional questions with sub-question hierarchies |
| Source | <https://cs.stanford.edu/people/jcjohns/clevr/> | <https://cs.stanford.edu/people/dorarad/gqa/> | <https://github.com/vis-nlp/ChartQA>, files from <https://huggingface.co/datasets/ahmed-masry/ChartQA> | <https://agqa-decomp.cs.washington.edu/>, videos from <https://prior.allenai.org/projects/charades> |
| Licence | CC BY 4.0 (dataset page and `LICENSE.txt` of the archive) | not stated on the download page; see the dataset page | GPL-3.0 (dataset card) | not stated on the dataset pages; videos under the Charades terms |
| Version | v1.0 | questions v1.2 | Hub revision `af8b6f5c08c9` | AGQA 2.0 balanced with hierarchies |
| Examples | train 70,000 images / 699,989 questions; val 15,000 images / 149,991 questions; test 15,000 images | balanced and full question files per split; counts on the dataset page | train 28,299; val 1,920; test 2,500 | 2.27M balanced question-answer pairs (AGQA 2.0); hierarchies for the balanced train and test sets |
| Annotation archives | `CLEVR_v1.0_no_images.zip` (89.4 MB) | `questions1.2.zip` (1.50 GB), `sceneGraphs.zip` (44.8 MB) | `ChartQA Dataset.zip` (875 MB, includes the chart images) | `AGQA-decomp-hierarchies.zip` (447 MB), `csvs.zip` (50.6 MB) |
| Media archives (`--media`) | `CLEVR_v1.0.zip` (19.0 GB, images and annotations) | `images.zip` (21.8 GB) | - | `Charades_v1_480.zip` (16.3 GB) |

Layout after download:

```text
clevr/CLEVR_v1.0/questions/CLEVR_{train,val,test}_questions.json
clevr/CLEVR_v1.0/scenes/CLEVR_{train,val}_scenes.json
clevr/CLEVR_v1.0/images/{train,val,test}/            with --media
gqa/                                                 question files, scene graphs; images with --media
chartqa/ChartQA Dataset/{train,val,test}/            {split}_human.json, {split}_augmented.json, png/, annotations/, tables/
agqa/                                                hierarchies, CSV files; videos with --media
```

## CLEVR records (shipped)

`data/clevr/clevr_val_calibration.jsonl.gz` and `data/clevr/clevr_val_test.jsonl.gz` hold the CLEVR v1.0 validation questions that the repository's CLEVR grammar (`qcv/grammars/clevr.yaml`) compiles: 15,723 of the 149,991 validation questions (14,031 existence questions and 1,692 counting questions over 10,620 images). Each line is a raw record as consumed by `qcv compile`: the question, its answer, the image path, and `gold` answers of the auxiliary variables (object counts and existence) computed from the CLEVR scene graphs and functional programs. The two files are this repository's default partition: a record goes to the calibration file when a hash of the seed `0` and its `instance_id` falls into the first 20%.

| File | Records | Bytes | SHA-256 |
|---|---|---|---|
| `clevr/clevr_val_calibration.jsonl.gz` | 3,148 | 77,200 | `08d92e8783ebf87768eaf8b15ee93695982489731be7daf00dac90a0244e274a` |
| `clevr/clevr_val_test.jsonl.gz` | 12,575 | 284,599 | `27e889f128448880612308fc5ee4a4a79786fbf7c331c53568bd9c2c7f306dec` |

Produced from `CLEVR_v1.0_no_images.zip` (89,363,837 bytes, SHA-256 `d4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4`) with

```bash
python scripts/download_data.py clevr
python scripts/build_clevr_slice.py        # convert_clevr.py --split val, then split_records.py --calibration-fraction 0.2 --seed 0
```

`clevr/MANIFEST.json` records the source archive, the grammar checksum, the selection counts, the partition settings, the commands and the checksums of both files. CLEVR is © 2017 Facebook, Inc. and licensed under CC BY 4.0; `clevr/LICENSE.txt` is the licence text from the archive and `clevr/ATTRIBUTION.md` states the source and the changes made. The records reference images as `data/clevr/CLEVR_v1.0/images/val/<file>`; the images are fetched with `python scripts/download_data.py clevr --media`.

## From a benchmark to QCV records

1. Convert to raw records (one JSON object per line: `instance_id`, `task`, `question`, `answer`, `media`, `gold`).
   * CLEVR: `python scripts/convert_clevr.py --split val --drop-incomplete-gold` reads `data/clevr/CLEVR_v1.0/` (fetching the annotation archive if it is absent), keeps the questions matched by the CLEVR grammar, derives the gold answers of the auxiliary variables from the scene graphs and writes `data/clevr/clevr_val_raw.jsonl`.
   * GQA, ChartQA, AGQA-Decomp: `python scripts/convert_records.py` maps the question files to raw records; gold answers of the auxiliary variables are supplied with `--gold-sidecar`. Example for GQA:
     `python scripts/convert_records.py --task gqa --input gqa/val_balanced_questions.json --question-field question --answer-field answer --media-field imageId --media-suffix .jpg --media-root gqa/images`
2. Split into calibration and test records: `python scripts/split_records.py --input data/clevr/clevr_val_raw.jsonl --calibration-fraction 0.2 --seed 0` (the assignment is a hash of the seed and the instance id; `.jsonl.gz` outputs are written compressed).
3. Compile, score, calibrate and evaluate as described in the main README.

The question templates and answer alphabets of every task are the grammars in `qcv/grammars/` (`clevr.yaml`, `gqa.yaml`, `chartqa.yaml`, `agqa.yaml`). Numeric alphabets can be frozen from calibration records with `scripts/fit_alphabets.py`.

## Synthetic scenes (shipped)

`data/synthetic/` holds the records used by `qcv demo`, by the tests and by the runs in `results/`. They are produced by the simulator in `qcv/synthetic.py` for the grammar `qcv/grammars/synthetic.yaml`:

```bash
qcv synth --output-dir data/synthetic        # seed 0, 1500 calibration and 2000 test records
```

Each record is a question about counts or spatial relations of two named objects together with a latent scene state (`gold`: one jointly valid answer tuple of the compiled pool, drawn uniformly). The simulated VLM scores are not stored; `qcv score --backend synthetic` produces them from the scene states. Licence: MIT, as the repository.

| File | Records | Bytes | SHA-256 |
|---|---|---|---|
| `synthetic/calibration_raw.jsonl` | 1,500 | 653,235 | `d0cd21f7852bc00bfb13629a4bcb9430a0001508f525669638ba6396fd157a88` |
| `synthetic/test_raw.jsonl` | 2,000 | 842,843 | `86ccb42f5cbb9bb51a8d528dc49ca1f01963a46cfb63ea6eb29ccf25312037f7` |

`synthetic/MANIFEST.json` stores the generator settings and the same checksums; `qcv demo` uses these files when its seed and sizes match the manifest and generates the records otherwise.
