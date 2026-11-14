"""Upstream sources of the benchmarks used in the paper.

Sizes are the ``Content-Length`` values reported by the servers on 2026-11-13.
``media`` archives hold images or videos and are fetched only on request.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteFile:
    url: str
    size_bytes: int
    extract: bool = True
    """Unpack the archive into the dataset directory after downloading."""
    media: bool = False
    """Images or videos; skipped unless media is requested."""
    filename: str | None = None

    @property
    def name(self) -> str:
        return self.filename or self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class DatasetSource:
    key: str
    title: str
    homepage: str
    licence: str
    version: str
    files: tuple[RemoteFile, ...]
    notes: str = ""


SOURCES: dict[str, DatasetSource] = {
    source.key: source
    for source in (
        DatasetSource(
            key="clevr",
            title="CLEVR v1.0",
            homepage="https://cs.stanford.edu/people/jcjohns/clevr/",
            licence="CC BY 4.0",
            version="v1.0",
            files=(
                RemoteFile(
                    "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip", 89_363_837
                ),
                RemoteFile(
                    "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
                    19_021_600_724,
                    media=True,
                ),
            ),
            notes="questions, scene graphs and functional programs; the full archive adds images",
        ),
        DatasetSource(
            key="gqa",
            title="GQA v1.2",
            homepage="https://cs.stanford.edu/people/dorarad/gqa/",
            licence="see the dataset page",
            version="questions v1.2",
            files=(
                RemoteFile(
                    "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip", 1_498_616_372
                ),
                RemoteFile(
                    "https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip", 44_824_862
                ),
                RemoteFile(
                    "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
                    21_817_965_542,
                    media=True,
                ),
            ),
            notes="balanced and all questions, scene graphs; images in a separate archive",
        ),
        DatasetSource(
            key="chartqa",
            title="ChartQA",
            homepage="https://github.com/vis-nlp/ChartQA",
            licence="GPL-3.0",
            version="Hugging Face ahmed-masry/ChartQA, revision af8b6f5c08c9",
            files=(
                RemoteFile(
                    "https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/"
                    "af8b6f5c08c95085271561c2a3f9d15f2b5a9031/ChartQA%20Dataset.zip",
                    875_370_872,
                    filename="ChartQA_Dataset.zip",
                ),
            ),
            notes="chart images, question files, annotations and data tables in one archive",
        ),
        DatasetSource(
            key="agqa",
            title="AGQA-Decomp (AGQA 2.0 balanced with question hierarchies)",
            homepage="https://agqa-decomp.cs.washington.edu/",
            licence="see the dataset page; videos under the Charades terms",
            version="AGQA 2.0 balanced",
            files=(
                RemoteFile(
                    "https://agqa-decomp.cs.washington.edu/data/AGQA-decomp-hierarchies.zip",
                    447_410_320,
                ),
                RemoteFile("https://agqa-decomp.cs.washington.edu/data/csvs.zip", 50_638_394),
                RemoteFile(
                    "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/"
                    "Charades_v1_480.zip",
                    16_339_546_533,
                    media=True,
                ),
            ),
            notes="question hierarchies and train/test CSVs; videos are the Charades 480p release",
        ),
    )
}
