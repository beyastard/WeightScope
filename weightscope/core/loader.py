"""
WeightScope - Model Loader

Handles local and HuggingFace-hosted model loading with memory-safety checks.
Supports both single-file models (model.safetensors) and sharded models
(model-00001-of-00003.safetensors, …).

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

import json
import re
import socket
import struct
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from huggingface_hub import model_info, hf_hub_download, list_repo_files

from ..config import MEMORY_SAFETY_THRESHOLD, MODELS_DIR, SUPPORTED_DTYPES
from ..utils  import (
    sanitize_model_name, compute_file_hash,
    get_available_ram_gb, get_total_ram_gb,
    format_number, ensure_dir,
)


# ─── Shard discovery ─────────────────────────────────────────────────────────

def find_safetensors_shards(model_dir: Path) -> List[Path]:
    """
    Return an ordered list of all .safetensors shards in *model_dir*.

    Priority / detection order
    --------------------------
    1. ``model.safetensors``           - single-file model (most common)
    2. ``model-NNNNN-of-MMMMM.safetensors`` - HF standard sharded naming
    3. Any ``*.safetensors`` files     - fallback for non-standard naming

    Sharded files are sorted numerically by shard index so they are always
    processed in the correct order.
    """
    # 1. Single file
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]

    # 2. Standard HF shard pattern
    shard_pattern = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$", re.IGNORECASE)
    shards = sorted(
        (p for p in model_dir.iterdir() if shard_pattern.match(p.name)),
        key=lambda p: int(shard_pattern.match(p.name).group(1)),
    )
    if shards:
        return shards

    # 3. Any *.safetensors (sorted alphabetically for reproducibility)
    fallback = sorted(model_dir.glob("*.safetensors"))
    return fallback


def _st_header_param_count(path: Path) -> int:
    """
    Count parameters by reading only the JSON header of a safetensors file.
    Does NOT load any tensor data into memory.
    Works for all dtypes including BF16/FP8 (no numpy involvement).
    """
    total = 0
    try:
        with open(path, "rb") as fh:
            hdr_len  = struct.unpack("<Q", fh.read(8))[0]
            hdr_json = json.loads(fh.read(hdr_len).rstrip(b" \x00"))

        for name, info in hdr_json.items():
            if name == "__metadata__":
                continue
            
            dtype = info.get("dtype", "")
            shape = info.get("shape", [])
            if not shape:
                continue
            
            n_elems = 1
            for d in shape:
                n_elems *= d
            
            # INT4 stores 2 values per byte
            if dtype in ("I4", "U4"):
                n_elems *= 2
            
            total += n_elems
    except Exception:
        pass
    
    return total


class ModelLoader:
    """
    Responsible for locating, downloading, and pre-checking model files.

    After a successful ``load_*`` call:
        - ``current_model_paths`` → ordered list of .safetensors shard Paths
        - ``current_model_id``    → human-readable identifier
        - ``config_data``         → parsed config.json (empty dict if absent)
        - ``is_remote``           → True when fetched from HuggingFace
        - ``shard_count``         → number of shards (1 for single-file models)
    """

    def __init__(self) -> None:
        self.current_model_paths: List[Path]       = []
        self.current_model_id:    Optional[str]    = None
        self.is_remote:           bool             = False
        self.config_data:         Dict[str, Any]   = {}

    @property
    def current_model_path(self) -> Optional[Path]:
        """Backwards-compatible single-path accessor (first shard)."""
        return self.current_model_paths[0] if self.current_model_paths else None

    @property
    def shard_count(self) -> int:
        return len(self.current_model_paths)

    # ─── Connectivity ─────────────────────────────────────────────────────────

    @staticmethod
    def check_connection() -> bool:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            return False

    # ─── Parameter-count helpers ──────────────────────────────────────────────

    @staticmethod
    def _count_params_from_shards(paths: List[Path]) -> int:
        """Sum parameter counts across all shards via header-only reads."""
        return sum(_st_header_param_count(p) for p in paths)

    @staticmethod
    def _count_params_from_config(config_path: Path) -> Optional[int]:
        if not config_path.exists():
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
            
            for key in ("num_parameters", "n_params", "parameter_count"):
                if key in config:
                    return int(config[key])
            
            if "hidden_size" in config and "num_hidden_layers" in config:
                h = config["hidden_size"]
                l = config["num_hidden_layers"]
                return int(h * h * l * 12)
        except Exception:
            pass
        
        return None

    @staticmethod
    def _count_params_from_hf(model_id: str) -> Optional[int]:
        try:
            info = model_info(model_id, timeout=10)
            if hasattr(info, "safetensors") and info.safetensors:
                return info.safetensors.get("total")
            
            if hasattr(info, "config") and info.config:
                return info.config.get("num_parameters")
        except Exception:
            pass
        return None

    # ─── Memory estimation ────────────────────────────────────────────────────

    @staticmethod
    def estimate_memory(param_count: int) -> Dict[str, Any]:
        """
        Estimate analysis RAM requirement.
        The streaming DuckDB engine holds at most one tensor at a time, so the
        peak overhead is roughly the size of the largest tensor (unknown without
        loading).  We use 0.5× param_count × 4 bytes as a conservative floor.
        """
        # Analysis peak: ~0.5 bytes per param (one tensor in RAM + DuckDB buffer)
        estimated_bytes = param_count * 0.5
        estimated_gb    = estimated_bytes / (1024 ** 3)
        available_gb    = get_available_ram_gb()

        safe_to_load    = estimated_gb < (available_gb * MEMORY_SAFETY_THRESHOLD)
        warning_level   = "safe"
        warning_message = ""

        if not safe_to_load:
            if estimated_gb > available_gb:
                warning_level   = "critical"
                warning_message = (
                    f"❌ CRITICAL: Estimated {estimated_gb:.1f} GB exceeds "
                    f"available {available_gb:.1f} GB."
                )
            else:
                warning_level   = "warning"
                warning_message = (
                    f"⚠️ WARNING: Estimated {estimated_gb:.1f} GB is "
                    f"{estimated_gb / available_gb * 100:.0f}% of available RAM."
                )
        elif estimated_gb > 8:
            warning_level   = "caution"
            warning_message = f"⚠️ CAUTION: Large model (~{estimated_gb:.1f} GB analysis footprint)."

        return {
            "param_count":     param_count,
            "estimated_gb":    round(estimated_gb, 2),
            "available_gb":    round(available_gb, 2),
            "total_gb":        round(get_total_ram_gb(), 2),
            "safe_to_load":    safe_to_load,
            "warning_level":   warning_level,
            "warning_message": warning_message,
        }

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {}

    def _resolve_param_count(
        self,
        shards: List[Path],
        config_path: Optional[Path] = None,
        hf_model_id: Optional[str] = None,
    ) -> int:
        """
        Resolve parameter count across all shards.
        Priority: shard headers > HF API > config.json
        """
        count = self._count_params_from_shards(shards)
        if count:
            return count
        
        if hf_model_id:
            count = self._count_params_from_hf(hf_model_id) or 0
        
        if not count and config_path:
            count = self._count_params_from_config(Path(config_path)) or 0
        
        return count

    # ─── Public load API ──────────────────────────────────────────────────────

    def load_local_model(self, model_dir: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Load a model from *model_dir*, auto-detecting single or sharded layout.

        Returns (success, message, memory_estimate_or_None).
        """
        model_path  = Path(model_dir)
        config_file = model_path / "config.json"

        if not model_path.is_dir():
            return False, f"❌ Directory not found: {model_dir}", None

        shards = find_safetensors_shards(model_path)
        if not shards:
            return False, f"❌ No .safetensors files found in {model_dir}", None

        param_count = self._resolve_param_count(shards, config_file)
        if not param_count:
            return False, "❌ Could not determine parameter count", None

        mem = self.estimate_memory(param_count)
        if not mem["safe_to_load"]:
            return False, mem["warning_message"], None

        self.current_model_paths = shards
        self.current_model_id    = model_path.name
        self.is_remote           = False
        self.config_data         = self._load_config(config_file)

        shard_info = (
            f"{len(shards)} shard{'s' if len(shards) > 1 else ''}"
            if len(shards) > 1
            else "single file"
        )
        return (
            True,
            f"✅ Loaded: {model_path.name}  ({format_number(param_count)} params, {shard_info})",
            mem,
        )

    def load_remote_model(
        self, model_id: str, cache_dir: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Download a model from HuggingFace Hub (single or sharded).

        Inspects the repo file list to discover all shards before downloading.
        Returns (success, message, memory_estimate_or_None).
        """
        if not self.check_connection():
            return False, "❌ No internet connection. Use local mode.", None

        try:
            model_info(model_id, timeout=10)
        except Exception as exc:
            return False, f"❌ Model not found: {exc}", None

        cache_root = Path(cache_dir) if cache_dir else MODELS_DIR
        cache_path = cache_root / sanitize_model_name(model_id)
        ensure_dir(cache_path)

        # ── Discover shard filenames from the repo ────────────────────────────
        try:
            all_files = list(list_repo_files(model_id))
        except Exception as exc:
            return False, f"❌ Could not list repo files: {exc}", None

        # Collect candidate shard filenames in priority order
        shard_filenames: List[str] = []

        # Priority 1: standard sharded pattern
        shard_re = re.compile(r"^model-(\d+)-of-(\d+)\.safetensors$", re.IGNORECASE)
        sharded = sorted(
            (f for f in all_files if shard_re.match(f)),
            key=lambda f: int(shard_re.match(f).group(1)),
        )
        if sharded:
            shard_filenames = sharded

        # Priority 2: single file
        elif "model.safetensors" in all_files:
            shard_filenames = ["model.safetensors"]

        # Priority 3: any .safetensors
        else:
            shard_filenames = sorted(f for f in all_files if f.endswith(".safetensors"))

        if not shard_filenames:
            return False, "❌ No .safetensors files found in this repo", None

        # ── Download all shards ───────────────────────────────────────────────
        downloaded: List[Path] = []
        for filename in shard_filenames:
            try:
                local = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    cache_dir=str(cache_path.parent),
                    force_download=False,
                )
                downloaded.append(Path(local))
            except Exception as exc:
                return False, f"❌ Download failed for {filename}: {exc}", None

        # ── config.json ───────────────────────────────────────────────────────
        config_path = None
        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                cache_dir=str(cache_path.parent),
                force_download=False,
            )
        except Exception:
            pass

        param_count = self._resolve_param_count(downloaded, config_path, model_id)
        if not param_count:
            return False, "❌ Could not determine parameter count", None

        mem = self.estimate_memory(param_count)
        if not mem["safe_to_load"]:
            return False, mem["warning_message"], None

        self.current_model_paths = downloaded
        self.current_model_id    = model_id
        self.is_remote           = True
        self.config_data         = self._load_config(config_path)

        shard_info = (
            f"{len(downloaded)} shard{'s' if len(downloaded) > 1 else ''}"
            if len(downloaded) > 1
            else "single file"
        )
        return (
            True,
            f"✅ Loaded: {model_id}  ({format_number(param_count)} params, {shard_info})",
            mem,
        )
