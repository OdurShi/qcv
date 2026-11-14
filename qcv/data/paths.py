"""Location of the data directory.

Resolution order: an explicit argument, the ``QCV_DATA_DIR`` environment
variable, ``./data`` in the current working directory (returned as the relative
path ``data`` so that paths written into records stay portable), and finally
the ``data`` directory next to the package source (a source checkout).
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_VARIABLE = "QCV_DATA_DIR"


def default_data_dir(explicit: str | Path | None = None) -> Path:
    """Return the data directory to use; it does not have to exist yet."""
    if explicit is not None:
        return Path(explicit)
    configured = os.environ.get(DATA_DIR_VARIABLE)
    if configured:
        return Path(configured)
    local = Path("data")
    if local.is_dir():
        return local
    checkout = Path(__file__).resolve().parents[2] / "data"
    return checkout if checkout.is_dir() else local
