"""Download benchmark archives into the ``data/`` layout and record what was fetched."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .sources import SOURCES, DatasetSource, RemoteFile

logger = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20
RECORD_NAME = "DOWNLOAD.json"
"""Per-dataset record of the fetched archives (URL, size, SHA-256); not tracked by git."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_context() -> ssl.SSLContext:
    """Certificate verification with the best trust store available.

    Uses the operating-system store through ``truststore`` or the ``certifi``
    bundle when one of them is installed, otherwise Python's default.
    """
    try:
        import truststore
    except ImportError:
        pass
    else:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _fetch_with_curl(url: str, partial: Path) -> None:
    command = ["curl", "--fail", "--location", "--silent", "--show-error"]
    subprocess.run([*command, "--output", str(partial), url], check=True)


def fetch(url: str, destination: Path) -> Path:
    """Stream ``url`` to ``destination`` (written atomically through a ``.part`` file).

    If Python cannot verify the server certificate because it has no CA bundle
    (common for standalone Python builds) and ``curl`` is installed, the file is
    fetched with ``curl`` instead; verification is never disabled.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context()))
    try:
        with opener.open(url) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, CHUNK_BYTES)
    except urllib.error.URLError as exc:
        unverifiable = isinstance(exc.reason, ssl.SSLCertVerificationError)
        if not unverifiable or shutil.which("curl") is None:
            raise
        logger.warning("Python has no usable CA bundle; fetching %s with curl", url)
        _fetch_with_curl(url, partial)
    partial.replace(destination)
    return destination


def _load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / RECORD_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}}


def download_dataset(
    source: DatasetSource,
    data_dir: Path,
    include_media: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch the archives of one dataset into ``<data_dir>/<key>/`` and unpack them.

    Archives are kept in ``<data_dir>/_downloads/``. A file that is already
    listed in the dataset's manifest with the same URL is not fetched again
    unless ``force`` is set. Returns the updated record (``DOWNLOAD.json``) with
    the URL, size and SHA-256 of every fetched archive.
    """
    target = data_dir / source.key
    target.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(target)
    manifest.update(
        {
            "dataset": source.title,
            "homepage": source.homepage,
            "licence": source.licence,
            "version": source.version,
        }
    )
    for remote in source.files:
        if remote.media and not include_media:
            continue
        known = manifest["files"].get(remote.name)
        if known and known.get("url") == remote.url and not force:
            logger.info("%s is already present, skipping", remote.name)
            continue
        archive = data_dir / "_downloads" / remote.name
        logger.info("downloading %s (%.1f MB)", remote.url, remote.size_bytes / 1e6)
        fetch(remote.url, archive)
        if remote.extract and zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(target)
        manifest["files"][remote.name] = {
            "url": remote.url,
            "bytes": archive.stat().st_size,
            "sha256": sha256_of(archive),
            "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extracted": bool(remote.extract),
        }
        (target / RECORD_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    return manifest


def ensure_file(path: Path, dataset: str, data_dir: Path) -> Path:
    """Return ``path``; fetch the dataset's non-media archives first if it is absent."""
    if not path.exists():
        logger.warning("%s not found; downloading the %s archives", path, dataset)
        download_dataset(SOURCES[dataset], data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} is not part of the downloaded {dataset} files")
    return path


def describe(source: DatasetSource, include_media: bool = False) -> list[RemoteFile]:
    """Files that ``download_dataset`` would fetch for these options."""
    return [remote for remote in source.files if include_media or not remote.media]
