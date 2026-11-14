"""Download the benchmarks used in the paper into the ``data/`` layout.

    python scripts/download_data.py --list
    python scripts/download_data.py clevr
    python scripts/download_data.py gqa chartqa agqa
    python scripts/download_data.py clevr --media      # also images / videos

Question and annotation archives are fetched by default; image and video
archives (tens of GB) only with ``--media``. Every fetched archive is recorded
with its URL, size and SHA-256 in ``data/<dataset>/DOWNLOAD.json``.
"""

from __future__ import annotations

import argparse
import logging

from qcv.data import default_data_dir
from qcv.data.download import describe, download_dataset
from qcv.data.sources import SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help=f"datasets to fetch: {', '.join(SOURCES)}")
    parser.add_argument("--data-dir", help="target directory (default: $QCV_DATA_DIR or ./data)")
    parser.add_argument("--media", action="store_true", help="also fetch images / videos")
    parser.add_argument("--force", action="store_true", help="fetch again even if recorded")
    parser.add_argument("--list", action="store_true", help="print the sources and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_dir = default_data_dir(args.data_dir)
    unknown = [key for key in args.datasets if key not in SOURCES]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; choose from {list(SOURCES)}")

    if args.list or not args.datasets:
        for source in SOURCES.values():
            print(f"{source.key}: {source.title} | licence: {source.licence} | {source.homepage}")
            for remote in source.files:
                kind = "media" if remote.media else "annotations"
                print(f"    {remote.size_bytes / 1e6:10.1f} MB  {kind:<11}  {remote.url}")
        return
    for key in args.datasets:
        source = SOURCES[key]
        planned = describe(source, include_media=args.media)
        total = sum(remote.size_bytes for remote in planned) / 1e6
        print(f"{key}: fetching {len(planned)} archive(s), {total:.1f} MB, into {data_dir / key}")
        manifest = download_dataset(source, data_dir, include_media=args.media, force=args.force)
        for name, entry in manifest["files"].items():
            print(f"    {name}  {entry['bytes']} bytes  sha256 {entry['sha256']}")


if __name__ == "__main__":
    main()
