"""
WeightScope - Weight Analysis Engine

Responsible for loading tensors, building the per-unique-value DataFrame that
every downstream analysis tab operates on, and running all quantitative
simulations (quantization, pruning, clip-normalize, low-count removal).

Memory model
------------
Analysis is **streaming and memory-bounded**.  Tensors are processed one at a
time; per-tensor unique counts are appended to a temporary DuckDB database.
After all tensors are processed, DuckDB performs a single aggregation in C++
(GROUP BY key, SUM count) that is far faster than a Python-side merge.
Peak RAM is bounded by the largest single tensor, not the full model size.

bfloat16 / FP8 note
--------------------
``safe_open(..., framework="np")`` calls numpy internally when materializing a
tensor.  NumPy does not recognize 'bfloat16' as a native dtype and raises
``TypeError: data type 'bfloat16' not understood`` before our code can see the
array.  FP8 variants exhibit the same issue on older NumPy builds.

The workaround is a **hybrid read strategy**:

1. Parse the safetensors JSON header once, up-front, to get each tensor's
   dtype string, shape, and byte offsets.
2. For dtypes that numpy handles (F32, F16, I8, U8, …) use the normal
   ``safe_open`` / ``get_tensor()`` path — fast and zero-copy.
3. For dtypes that crash numpy (BF16, F8_E4M3, F8_E5M2) read the raw bytes
   directly from the file and convert to float32 manually.

Supported dtypes: FP32 · FP16 · BF16 · FP8(e4m3/e5m2) · INT8 · UINT8 · INT4

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
import struct
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import gradio as gr
import numpy as np
import pandas as pd
from safetensors import safe_open

from ..config import SUPPORTED_DTYPES, ANALYSIS_CHUNK_SIZE, ANALYSIS_TEMP_DIR
from ..utils  import compute_file_hash, format_number, ensure_dir


# ─── Safetensors dtype constants ─────────────────────────────────────────────
# Dtype strings as they appear in the safetensors JSON header.
# BF16 and FP8 variants cannot be materialized through numpy; they require the
# raw-byte fallback path.

# safe_open-compatible: numpy can handle these directly
_ST_NP_COMPAT = {"F32", "F16", "I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64", "F64"}

# Require raw-byte read: numpy rejects these dtype names
_ST_RAW_READ  = {"BF16", "F8_E4M3", "F8_E5M2"}

# Our internal WeightScope label for each safetensors dtype string
_ST_LABEL = {
    "F32":    "float32",
    "F16":    "float16",
    "BF16":   "bfloat16",
    "F8_E4M3":"float8_e4m3fn",
    "F8_E5M2":"float8_e5m2",
    "I8":     "int8",
    "U8":     "uint8",
}

# Safetensors dtype string → element size in bytes (for raw reads)
_ST_ELEM_BYTES = {
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


# ─── Safetensors header parser ────────────────────────────────────────────────

def _parse_st_header(path: Path) -> Tuple[Dict, int]:
    """
    Read and parse the safetensors JSON header without loading any tensor data.

    Returns
    -------
    (header_dict, data_section_start_byte)
        ``header_dict`` maps tensor names to their metadata dicts
        (``dtype``, ``shape``, ``data_offsets``).
        ``data_section_start_byte`` is the file offset where tensor data begins.
    """
    with open(path, "rb") as fh:
        hdr_len  = struct.unpack("<Q", fh.read(8))[0]
        hdr_json = json.loads(fh.read(hdr_len).rstrip(b" \x00"))
    data_start = 8 + hdr_len
    
    # Strip the optional __metadata__ pseudo-key
    tensors = {k: v for k, v in hdr_json.items() if k != "__metadata__"}
    return tensors, data_start


# ─── Per-tensor conversion helpers ────────────────────────────────────────────

def _unpack_int4(raw: np.ndarray) -> np.ndarray:
    """
    Unpack a UINT8 array storing two INT4 values per byte.
    Lower nibble first (GGUF / llama.cpp convention).
    Returns a sign-extended float32 array twice the length of *raw*.
    """
    lo = (raw & 0x0F).astype(np.int8)
    hi = ((raw >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    return np.stack([lo, hi], axis=-1).ravel().astype(np.float32)


def _np_tensor_to_keys(tensor: np.ndarray, st_dtype: str) -> Optional[Tuple[np.ndarray, str]]:
    """
    Convert a numpy tensor (already materialized by safe_open) into
    ``(uint32_keys, dtype_label)``.

    Called only for dtypes in ``_ST_NP_COMPAT``.
    Returns None if the dtype is not one WeightScope analyzes.
    """
    label = _ST_LABEL.get(st_dtype)
    if label is None:
        return None

    if st_dtype == "F32":
        keys = tensor.ravel().view(np.uint32).copy()
        return keys, label

    if st_dtype == "F16":
        fvals = tensor.ravel().astype(np.float32)
        return fvals.view(np.uint32).copy(), label

    if st_dtype in ("I8", "U8"):
        fvals = tensor.ravel().astype(np.float32)
        return fvals.view(np.uint32).copy(), label

    # INT4 is packed into U8 storage; the dtype in the header is typically
    # reported as U8 by the writing framework.  If somehow passed here, handle.
    return None


def _raw_bytes_to_keys(
    raw: bytes,
    st_dtype: str,
    shape: List[int],
) -> Optional[Tuple[np.ndarray, str]]:
    """
    Convert raw tensor bytes (for BF16/FP8 tensors that numpy can't handle)
    into ``(uint32_keys, dtype_label)``.

    Called only for dtypes in ``_ST_RAW_READ``.
    """
    label = _ST_LABEL.get(st_dtype)
    if label is None:
        return None

    if st_dtype == "BF16":
        # Each bfloat16 value is 2 bytes.  Interpret as uint16, then shift
        # left 16 bits to sit in the float32 exponent/sign position.
        # The resulting float32 bit-pattern is an exact (zero-padded mantissa)
        # representation of the original bfloat16 value.
        u16   = np.frombuffer(raw, dtype=np.uint16)
        u32   = u16.astype(np.uint32) << 16
        # u32 IS our key array — viewing it as float32 gives the weight value
        return u32.copy(), label

    if st_dtype in ("F8_E4M3", "F8_E5M2"):
        # Upcast 8-bit floats to float32; use the float32 bit-pattern as key.
        u8    = np.frombuffer(raw, dtype=np.uint8)
        fvals = u8.astype(np.float32)
        return fvals.view(np.uint32).copy(), label

    return None


# ─── Streaming frequency counter ─────────────────────────────────────────────

class _StreamingCounter:
    """
    Accumulates per-tensor unique-value counts into a temporary DuckDB file.

    Workflow
    --------
    1. For each tensor call ``feed(keys)`` — runs ``np.unique`` on that
       tensor's uint32 keys and bulk-inserts (key, count) pairs into DuckDB.
    2. After all tensors call ``finalize()`` — DuckDB performs a single
       ``GROUP BY k ORDER BY k`` aggregation in C++ and returns a DataFrame.
       The temp DB file is deleted automatically.

    Peak RAM
    --------
    ``np.unique`` on one tensor at a time is the only large allocation.
    """

    def __init__(self, chunk_size: int, temp_dir: Path) -> None:
        self._total_params = 0
        self._db_path      = temp_dir / f"ws_{uuid.uuid4().hex}.duckdb"
        self._con          = duckdb.connect(str(self._db_path))
        self._con.execute("CREATE TABLE weights (k UINTEGER NOT NULL, c UBIGINT NOT NULL)")

    def feed(self, keys: np.ndarray) -> None:
        """Compute unique counts for *keys* (one tensor) and store in DuckDB."""
        self._total_params += len(keys)
        unique_k, counts = np.unique(keys, return_counts=True)
        df = pd.DataFrame({
            "k": unique_k.astype(np.uint32),
            "c": counts.astype(np.uint64),
        })
        self._con.append("weights", df)

    def finalize(self) -> pd.DataFrame:
        """
        Aggregate all (key, count) rows, recover float32 values, return DataFrame.
        Deletes the temp DuckDB file.
        """
        res = self._con.execute("SELECT k, SUM(c) AS count FROM weights GROUP BY k ORDER BY k").df()
        self._con.close()
        self._db_path.unlink(missing_ok=True)

        if res.empty:
            return pd.DataFrame(columns=["value", "count", "bit_pattern"])

        keys_u32 = np.ascontiguousarray(res["k"].values, dtype=np.uint32)
        fvals    = keys_u32.view(np.float32).copy()

        return pd.DataFrame({
            "value":       fvals,
            "count":       res["count"].values.astype(np.int64),
            "bit_pattern": [f"0x{k:08X}" for k in keys_u32],
        })

    def cleanup(self) -> None:
        """Emergency cleanup — safe to call even if __init__ partially failed."""
        try:
            self._con.close()
        except Exception:
            pass
        try:
            self._db_path.unlink(missing_ok=True)
        except Exception:
            pass

    @property
    def total_params(self) -> int:
        return self._total_params


# ─── Main analyzer ────────────────────────────────────────────────────────────

class WeightAnalyzer:
    """
    Core analysis engine for safetensors model weights.

    Typical usage::

        analyzer = WeightAnalyzer()
        ok, msg  = analyzer.analyze_model(Path("model.safetensors"))
        print(analyzer.df.head())
    """

    def __init__(self) -> None:
        self.df:             Optional[pd.DataFrame] = None
        self.model_hash:     Optional[str]          = None
        self.model_metadata: Optional[Dict]         = None

    def analyze_model(
        self,
        safetensors_paths: Union[Path, List[Path]],
        progress=gr.Progress(),
        chunk_size: Optional[int] = None,
        temp_dir:   Optional[Path] = None,
    ) -> Tuple[bool, str]:
        """
        Stream all supported tensors through the DuckDB frequency counter.

        Accepts a single Path **or** an ordered list of shard Paths.  Shards
        are processed sequentially; all share the same ``_StreamingCounter`` so
        the final DuckDB aggregation sees every tensor across every shard.

        Uses a hybrid read strategy per shard:
        - Numpy-safe dtypes (F32, F16, I8, U8, …) → ``safe_open`` / ``get_tensor()``
        - BF16 / FP8                               → raw byte read + manual conversion

        Returns (success, status_message).
        """
        # Normalize to list
        if isinstance(safetensors_paths, Path):
            safetensors_paths = [safetensors_paths]

        _tmp_dir = temp_dir or ANALYSIS_TEMP_DIR
        ensure_dir(_tmp_dir)

        counter: Optional[_StreamingCounter] = None

        try:
            # ── Composite hash across all shards (order-stable) ───────────────
            import hashlib
            h = hashlib.sha256()
            for p in safetensors_paths:
                h.update(compute_file_hash(p).encode())
            self.model_hash = h.hexdigest()

            # ── Count total tensors across all shards for progress reporting ──
            shard_headers: List[Tuple[Dict, int, Path]] = []
            total_tensors = 0
            for shard_path in safetensors_paths:
                hdr, data_start = _parse_st_header(shard_path)
                shard_headers.append((hdr, data_start, shard_path))
                total_tensors += len([k for k in hdr if k != "__metadata__"])

            tensor_info: List[Dict] = []
            skipped:     List[str]  = []
            counter = _StreamingCounter(chunk_size or ANALYSIS_CHUNK_SIZE, _tmp_dir)
            done    = 0

            # ── Iterate shards ────────────────────────────────────────────────
            for shard_idx, (st_header, data_start, shard_path) in enumerate(shard_headers):
                shard_label = shard_path.name
                tensor_keys = [k for k in st_header.keys() if k != "__metadata__"]

                np_compat_keys = [
                    k for k in tensor_keys
                    if st_header[k]["dtype"] in _ST_NP_COMPAT
                    and st_header[k]["dtype"] in _ST_LABEL
                ]
                raw_read_keys = [
                    k for k in tensor_keys
                    if st_header[k]["dtype"] in _ST_RAW_READ
                ]
                for k in tensor_keys:
                    if k not in np_compat_keys and k not in raw_read_keys:
                        skipped.append(f"{k} ({st_header[k]['dtype']}) in {shard_label}")

                # Step 2a: numpy-safe path
                if np_compat_keys:
                    with safe_open(str(shard_path), framework="np") as st:
                        for key in np_compat_keys:
                            progress(
                                done / total_tensors,
                                desc=f"[{shard_label}] {key} ({done+1}/{total_tensors})",
                            )
                            info   = st_header[key]
                            tensor = st.get_tensor(key)
                            result = _np_tensor_to_keys(tensor, info["dtype"])
                            done  += 1

                            if result is None:
                                skipped.append(f"{key} ({info['dtype']}) in {shard_label}")
                                del tensor
                                continue

                            keys_arr, label = result
                            counter.feed(keys_arr)
                            tensor_info.append({
                                "name":        key,
                                "shard":       shard_label,
                                "shape":       info["shape"],
                                "dtype":       info["dtype"],
                                "dtype_label": label,
                                "size":        int(np.prod(info["shape"])),
                            })
                            del keys_arr, tensor, result

                # Step 2b: raw-byte path for BF16/FP8
                if raw_read_keys:
                    with open(shard_path, "rb") as fh:
                        fh.seek(data_start)
                        data_region = fh.read()

                    for key in raw_read_keys:
                        progress(
                            done / total_tensors,
                            desc=f"[{shard_label}] {key} ({done+1}/{total_tensors})",
                        )
                        info      = st_header[key]
                        st_dtype  = info["dtype"]
                        s, e      = info["data_offsets"]
                        result    = _raw_bytes_to_keys(data_region[s:e], st_dtype, info["shape"])
                        done     += 1

                        if result is None:
                            skipped.append(f"{key} ({st_dtype}) in {shard_label}")
                            continue

                        keys_arr, label = result
                        counter.feed(keys_arr)
                        tensor_info.append({
                            "name":        key,
                            "shard":       shard_label,
                            "shape":       info["shape"],
                            "dtype":       st_dtype,
                            "dtype_label": label,
                            "size":        int(np.prod(info["shape"])),
                        })
                        del keys_arr, result

                    del data_region

            if not tensor_info:
                counter.cleanup()
                return False, "❌ No supported tensors found across all shards"

            progress(0.90, desc="Aggregating frequency table…")
            self.df = counter.finalize()
            counter  = None

            n_shards = len(safetensors_paths)
            progress(0.97, desc="Building metadata…")
            self.model_metadata = {
                "file_path":          str(safetensors_paths[0].parent),
                "shard_paths":        [str(p) for p in safetensors_paths],
                "shard_count":        n_shards,
                "file_hash":          self.model_hash,
                "total_parameters":   int(self.df["count"].sum()),
                "unique_patterns":    len(self.df),
                "tensor_count":       len(tensor_info),
                "skipped_tensors":    skipped,
                "tensors":            tensor_info,
                "analysis_timestamp": datetime.now().isoformat(),
                "dtypes_found":       list({t["dtype_label"] for t in tensor_info}),
            }

            progress(1.0, desc="Complete")
            shard_str = f"{n_shards} shard{'s' if n_shards > 1 else ''}"
            return (
                True,
                f"✅ Analyzed {format_number(self.model_metadata['total_parameters'])} "
                f"parameters, {format_number(len(self.df))} unique patterns "
                f"across {shard_str} ({len(skipped)} tensor(s) skipped)",
            )

        except Exception as exc:
            if counter is not None:
                counter.cleanup()
            import traceback
            tb = traceback.format_exc()
            return False, f"❌ Analysis failed: {exc}\n\nTraceback:\n{tb}"

    # ─── Compression simulations ──────────────────────────────────────────────

    def get_compression_analysis(self) -> Dict[str, Any]:
        if self.df is None:
            return {}
        
        unique_count = len(self.df)
        total_params = int(self.df["count"].sum())
        value_range  = float(self.df["value"].max() - self.df["value"].min())
        options: List[Dict] = []
        
        if unique_count <= 65_535:
            options.append({
                "method":       "uint16_index + lookup_table",
                "current_bits": 32, "new_bits": 16,
                "savings_mb":   round((32-16)/8 * total_params / (1024**2), 2),
                "description":  "Map each unique value to a 16-bit index",
            })
        
        if unique_count <= 255:
            options.append({
                "method":       "uint8_index + lookup_table",
                "current_bits": 32, "new_bits": 8,
                "savings_mb":   round((32-8)/8 * total_params / (1024**2), 2),
                "description":  "Map each unique value to an 8-bit index",
            })
        
        return {"unique_count": unique_count, "total_params": total_params,
                "value_range": round(value_range, 6), "options": options}

    def simulate_quantization(self, bits: int = 8, method: str = "mse") -> Dict[str, Any]:
        if self.df is None:
            return {"error": "No data loaded"}
        
        values = self.df["value"].values
        counts = self.df["count"].values
        v_min, v_max = float(values.min()), float(values.max())
        v_range = v_max - v_min
        levels  = 2 ** bits
        
        if v_range > 0:
            step      = v_range / levels
            quantized = np.round((values - v_min) / step) * step + v_min
        else:
            step, quantized = 0.0, values.copy()
        
        error = np.abs(values - quantized)
        return {
            "bits": bits, "method": method,
            "mse":       float(np.sum(error**2 * counts) / np.sum(counts)),
            "mae":       float(np.sum(error    * counts) / np.sum(counts)),
            "max_error": float(error.max()),
            "levels":    levels,
            "step_size": float(step),
            "range":     [v_min, v_max],
        }

    def simulate_low_count_removal(self, max_count: int = 4) -> Dict[str, Any]:
        if self.df is None:
            return {"error": "No data loaded"}
        
        total_params   = int(self.df["count"].sum())
        removable      = self.df[self.df["count"] <= max_count]
        removed_params = int(removable["count"].sum())
        removed_unique = len(removable)
        
        return {
            "removed_parameters":      removed_params,
            "removed_unique_patterns": removed_unique,
            "param_reduction_pct":     round(removed_params / total_params * 100, 3),
            "unique_reduction_pct":    round(removed_unique / len(self.df) * 100, 2),
            "remaining_params":        total_params - removed_params,
            "remaining_unique":        len(self.df) - removed_unique,
            "compression_gain":        round(
                (1 - (len(self.df) - removed_unique) / len(self.df)) * 100, 2),
        }

    def simulate_clipping_normalization(
        self, threshold: float, normalize_to: Tuple[float, float] = (-1.0, 1.0),
    ) -> Dict[str, Any]:
        if self.df is None:
            return {"error": "No data loaded"}
        
        values, counts  = self.df["value"].values, self.df["count"].values
        total           = float(counts.sum())
        tmin, tmax      = normalize_to
        t_range         = tmax - tmin
        c_range         = 2 * threshold
        clipped         = np.clip(values, -threshold, threshold)
        normalized      = (clipped + threshold) / c_range * t_range + tmin
        reconstructed   = (normalized - tmin) / t_range * c_range - threshold
        error           = values - reconstructed
        mse             = float(np.sum(error**2 * counts) / total)
        mae             = float(np.sum(np.abs(error) * counts) / total)
        sig_power       = float(np.sum(values**2 * counts) / total)
        snr_db          = 10 * np.log10(sig_power / mse) if mse > 0 else float("inf")
        clipped_mask    = np.abs(values) > threshold
        orig_range      = float(values.max() - values.min())
        bits_saved      = float(max(0, np.log2(orig_range / t_range))) if t_range > 0 and orig_range > 0 else 0.0
        
        return {
            "threshold": threshold, "target_range": list(normalize_to),
            "mse": mse, "mae": mae, "snr_db": snr_db,
            "clipped_parameters":    int(counts[clipped_mask].sum()),
            "clipped_unique_values": int(clipped_mask.sum()),
            "clipped_pct":           round(counts[clipped_mask].sum() / total * 100, 3),
            "unique_before":         len(values),
            "unique_after_clipping": len(np.unique(clipped)),
            "unique_reduction_pct":  round((1 - len(np.unique(clipped)) / len(values)) * 100, 2),
            "theoretical_bits_saved": round(bits_saved, 2),
            "compression_gain_pct":  round(bits_saved / 32 * 100, 2),
            "original_range":        [float(values.min()), float(values.max())],
            "clipped_range":         [-threshold, threshold],
            "normalized_range":      list(normalize_to),
        }

    def get_pruning_candidates(self, threshold: float = 1e-4) -> Dict[str, Any]:
        if self.df is None:
            return {}
        
        candidates   = self.df[np.abs(self.df["value"]) <= threshold].copy()
        total_params = int(self.df["count"].sum())
        prunable     = int(candidates["count"].sum())
        
        return {
            "threshold":           threshold,
            "prunable_parameters": prunable,
            "sparsity_pct":        round(prunable / total_params * 100, 4),
            "unique_candidates":   len(candidates),
            "candidates":          candidates.sort_values("count", ascending=False),
        }
