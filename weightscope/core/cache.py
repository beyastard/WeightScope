"""
WeightScope - Session Cache

Persists analysis results to disk so that re-loading the same model is instant.
Cache is keyed by model-ID and validated with the file's SHA-256 hash.

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

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import pandas as pd

from ..config import SAVE_STATE_DIR
from ..utils  import sanitize_model_name, ensure_dir


class SessionCache:
    """
    Disk-backed cache for weight analysis results.

    Layout::

        .save_state/
        └── <sanitized_model_id>/
            ├── analysis_state.parquet   ← the unique-value DataFrame
            ├── metadata.json            ← model metadata + file hash
            └── plots/                   ← (reserved for saved plot images)
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or SAVE_STATE_DIR
        ensure_dir(self.base_dir)

    # ─── Path helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, model_id: Union[str, None]) -> Path:
        return self.base_dir / sanitize_model_name(model_id)

    # ─── Public API ───────────────────────────────────────────────────────────

    def check_cache(self, model_id: Union[str, None], file_hash: str) -> bool:
        """
        Return True if a valid, hash-matching cache entry exists for *model_id*.
        """
        if not isinstance(model_id, str):
            return False

        cache_path    = self._cache_path(model_id)
        state_file    = cache_path / "analysis_state.parquet"
        metadata_file = cache_path / "metadata.json"

        if not (state_file.exists() and metadata_file.exists()):
            return False

        try:
            with open(metadata_file, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            return meta.get("file_hash") == file_hash
        except Exception:
            return False

    def save_state(
        self,
        model_id: str,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
    ) -> None:
        """Persist *df* and *metadata* to the cache directory for *model_id*."""
        if not isinstance(model_id, str):
            return

        cache_path = self._cache_path(model_id)
        ensure_dir(cache_path)
        ensure_dir(cache_path / "plots")

        # Drop internal helper columns that should not be serialised
        export_df = df.drop(columns=["bit_key"], errors="ignore")
        export_df.to_parquet(cache_path / "analysis_state.parquet", index=False)

        with open(cache_path / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def load_state(
        self, model_id: Union[str, None]
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Load a previously cached analysis.

        Returns (df, metadata) or (None, None) if no cache exists.
        """
        if not isinstance(model_id, str):
            return None, None

        cache_path    = self._cache_path(model_id)
        state_file    = cache_path / "analysis_state.parquet"
        metadata_file = cache_path / "metadata.json"

        if not (state_file.exists() and metadata_file.exists()):
            return None, None

        try:
            df = pd.read_parquet(state_file)
            with open(metadata_file, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            return df, metadata
        except Exception:
            return None, None

    def invalidate(self, model_id: Union[str, None]) -> None:
        """Delete the cached state for *model_id* (if it exists)."""
        import shutil
        if not isinstance(model_id, str):
            return
        
        cache_path = self._cache_path(model_id)
        if cache_path.exists():
            shutil.rmtree(cache_path)

    def list_cached_models(self) -> list[str]:
        """Return a list of model IDs that have cached analyses."""
        return [
            p.name.replace("--", "/")
            for p in self.base_dir.iterdir()
            if p.is_dir() and (p / "metadata.json").exists()
        ]

    # ─── Export ───────────────────────────────────────────────────────────────

    def export_data(
        self,
        model_id: Union[str, None],
        output_dir: str,
        fmt: str = "parquet",
    ) -> str:
        """
        Export cached analysis data to *output_dir* in the requested *fmt*.

        Supported formats: ``parquet``, ``csv``, ``json``.
        Returns a human-readable status string.
        """
        if not isinstance(model_id, str):
            return "❌ Invalid model ID"

        cache_path = self._cache_path(model_id)
        state_file = cache_path / "analysis_state.parquet"

        if not state_file.exists():
            return "❌ No analysis data to export — load and analyse a model first"

        out = Path(output_dir)
        ensure_dir(out)

        try:
            df          = pd.read_parquet(state_file)
            export_name = f"{sanitize_model_name(model_id)}_weights"

            if fmt == "parquet":
                dest = out / f"{export_name}.parquet"
                df.to_parquet(dest, index=False)
            elif fmt == "csv":
                dest = out / f"{export_name}.csv"
                df.to_csv(dest, index=False)
            elif fmt == "json":
                dest = out / f"{export_name}.json"
                df.to_json(dest, orient="records", indent=2)
            else:
                return f"❌ Unsupported format: {fmt}"

            return f"✅ Exported to {dest}"

        except Exception as exc:
            return f"❌ Export failed: {exc}"
