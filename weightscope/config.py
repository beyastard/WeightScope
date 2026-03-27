"""
WeightScope Configuration & Constants

All application-level settings in one place.
Override by setting environment variables or editing this file.

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

import os
import tempfile
from pathlib import Path

# ─── App Identity ─────────────────────────────────────────────────────────────
APP_NAME    = "WeightScope: SafeTensors Model Analyzer"
APP_VERSION = "0.2.1"

# ─── Paths ────────────────────────────────────────────────────────────────────
SAVE_STATE_DIR = Path(os.environ.get("WEIGHTSCOPE_CACHE_DIR",    ".save_state"))
MODELS_DIR     = Path(os.environ.get("WEIGHTSCOPE_MODELS_DIR",   "models"))
OUTPUT_DIR     = Path(os.environ.get("WEIGHTSCOPE_OUTPUT_DIR",   "output"))
PLUGINS_DIR    = Path(os.environ.get("WEIGHTSCOPE_PLUGINS_DIR",  "plugins"))

# ─── Analysis ─────────────────────────────────────────────────────────────────
DEFAULT_PRUNING_THRESHOLD = 1e-4
MAX_UNIQUE_FOR_PLOT       = 100_000
MEMORY_SAFETY_THRESHOLD   = 0.90    # fraction of available RAM

# ─── Streaming / memory-bounded analysis ──────────────────────────────────────
# Maximum number of individual weight *values* held in RAM at any one time
# during the streaming analysis pass.  Each slot needs ~8 bytes (uint32 key +
# int64 count in the in-memory merge dict), so:
#
#   4 000 000 entries ≈ 32 MB RAM  — safe on 8 GB systems
#   8 000 000 entries ≈ 64 MB RAM  — comfortable on 16 GB systems
#   16 000 000 entries ≈ 128 MB RAM — for 32 GB+ systems
#
# Lower the value to reduce peak RAM; raise it to reduce the number of disk
# flush operations (fewer flushes = faster analysis on large models).
ANALYSIS_CHUNK_SIZE = int(os.environ.get("WEIGHTSCOPE_CHUNK_SIZE", 4_000_000))

# Temporary work directory used by the streaming analysis engine.
# A small SQLite database is written here during analysis and deleted on
# completion.  Defaults to the OS temp dir.  Override to a drive with more
# free space, e.g.:
#   WEIGHTSCOPE_TEMP_DIR=/mnt/scratch python app.py
_env_tmp        = os.environ.get("WEIGHTSCOPE_TEMP_DIR", "").strip()
ANALYSIS_TEMP_DIR = Path(_env_tmp) if _env_tmp else Path(tempfile.gettempdir())

# ─── Supported dtypes ─────────────────────────────────────────────────────────
# Maps numpy dtype string → (view_dtype, bits_per_element)
SUPPORTED_DTYPES = {
    "float32":       ("uint32", 32),
    "float16":       ("uint16", 16),
    "bfloat16":      ("uint16", 16),  # same storage width as float16
    "float8_e4m3fn": ("uint8",   8),
    "float8_e5m2":   ("uint8",   8),
    "int8":          ("int8",    8),
    "uint8":         ("uint8",   8),
    "int4":          (None,      4),  # packed – handled specially
}

# ─── Server ───────────────────────────────────────────────────────────────────
SERVER_HOST = os.environ.get("WEIGHTSCOPE_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("WEIGHTSCOPE_PORT", "7860"))
