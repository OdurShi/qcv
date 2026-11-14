"""Crop the numbered figures of the paper from its PDF into ``assets/paper_figures/``.

The PDF is only read. Crop boxes are given in PDF points (72 per inch, origin at
the top-left corner of the page) and exclude the captions, the body text and
the page furniture. Requires PyMuPDF (``pip install pymupdf``), which is not a
dependency of the ``qcv`` package.

    python scripts/extract_paper_figures.py /path/to/paper.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

FIGURES = {
    # name: (page number starting at 1, (x0, y0, x1, y1))
    "fig01": (1, (317.0, 212.0, 560.0, 453.0)),
    "fig02": (3, (52.0, 52.0, 560.0, 238.0)),
    "fig03": (4, (54.0, 47.0, 558.0, 318.0)),
    "fig04": (7, (317.0, 266.0, 560.0, 534.0)),
    "fig05": (7, (52.0, 51.0, 560.0, 196.0)),
}


def extract(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    """Render every crop box of ``FIGURES`` to ``<output_dir>/<name>.png``."""
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with fitz.open(pdf_path) as document:
        for name, (page_number, box) in FIGURES.items():
            page = document[page_number - 1]
            pixmap = page.get_pixmap(clip=fitz.Rect(*box), dpi=dpi)
            target = output_dir / f"{name}.png"
            pixmap.save(target)
            written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", help="path to the paper PDF (read only)")
    parser.add_argument("--output-dir", default="assets/paper_figures")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    for path in extract(Path(args.pdf), Path(args.output_dir), args.dpi):
        print(f"{path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
