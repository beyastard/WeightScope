"""
WeightScope - General-purpose utility helpers.

---

Copyright (C) 2026 Bryan K Reinhart & BeySoft

This file is part of WeightScope.

WeightScope is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

WeightScope is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public
License along with WeightScope. If not, see <https://www.gnu.org/licenses/>.
"""

import hashlib
from pathlib import Path
from typing import Union

import psutil


# ─── Filesystem ───────────────────────────────────────────────────────────────

def sanitize_model_name(model_id: Union[str, None]) -> str:
    """Convert a HuggingFace model-ID to a filesystem-safe string."""
    if model_id is None:
        return "unknown_model"
    if not isinstance(model_id, str):
        model_id = str(model_id)
    return model_id.replace("/", "--")


def ensure_dir(path: Path) -> Path:
    """Create *path* (and any parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─── Hashing ──────────────────────────────────────────────────────────────────

def compute_file_hash(file_path: Path) -> str:
    """Return the SHA-256 hex-digest of *file_path* (chunked, memory-safe)."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ─── Memory ───────────────────────────────────────────────────────────────────

def get_available_ram_gb() -> float:
    """Return available system RAM in gigabytes."""
    return psutil.virtual_memory().available / (1024 ** 3)


def get_total_ram_gb() -> float:
    """Return total system RAM in gigabytes."""
    return psutil.virtual_memory().total / (1024 ** 3)


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_number(n: int) -> str:
    """Return *n* formatted with thousands separators, e.g. ``1,234,567``."""
    return f"{n:,}"
