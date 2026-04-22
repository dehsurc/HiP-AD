from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional


class GradNormCSVDumper:
    """Append-mode CSV writer used only on rank 0.

    Columns are inferred from the first appended row (order preserved via
    Python dict insertion order). If a pre-existing file is present and
    ``roll_existing`` is True, it is renamed to ``<stem>_<timestamp>.<ext>``
    before writing begins.
    """

    def __init__(
        self,
        path: str,
        rank: int = 0,
        roll_existing: bool = True,
    ) -> None:
        self._path = path
        self._rank = int(rank)
        self._active = self._rank == 0
        self._fieldnames: Optional[List[str]] = None

        if self._active and roll_existing and os.path.exists(path):
            stem, ext = os.path.splitext(path)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(path, f"{stem}_{stamp}{ext}")

    def append(self, row: Dict[str, object]) -> None:
        if not self._active:
            return
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        new_file = not os.path.exists(self._path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
