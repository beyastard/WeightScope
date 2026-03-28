"""
WeightScope Plugin - Layer-by-Layer Breakdown
=============================================

Breaks the global analysis down per tensor, showing each layer's mean,
standard deviation, sparsity %, min, max, and unique value count in a
sortable table.  A bar chart visualizes sparsity across all layers so
outlier tensors (collapsed layers, embedding tables, etc.) are immediately
obvious.

Requires: current_metadata (for the tensor list with sizes/shapes/dtypes)
          current_df       (for the global frequency table — used to cross-
                            reference which patterns belong to which layer)

Note: because WeightScope's frequency table is global (all tensors merged),
this plugin re-reads per-tensor statistics from the safetensors header and
the frequency table, computing weighted stats per layer from available data.
For exact per-layer distributions, individual tensor reloads would be needed;
this plugin uses the available metadata for fast zero-reload analysis.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from safetensors import safe_open

from weightscope.plugins.base import BasePlugin

# Dtypes that safe_open can materialize through numpy
_NP_COMPAT = {"F32", "F16", "I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64"}
_RAW_READ  = {"BF16", "F8_E4M3", "F8_E5M2"}


def _read_tensor_as_f32(shard_path: Path, tensor_name: str, st_dtype: str, data_offsets: list, data_start: int) -> Optional[np.ndarray]:
    """Load one tensor as float32, using the appropriate read path."""
    try:
        if st_dtype in _NP_COMPAT:
            with safe_open(str(shard_path), framework="np") as f:
                t = f.get_tensor(tensor_name)
            return t.ravel().astype(np.float32)

        if st_dtype == "BF16":
            with open(shard_path, "rb") as fh:
                fh.seek(data_start + data_offsets[0])
                raw = fh.read(data_offsets[1] - data_offsets[0])
            u16 = np.frombuffer(raw, dtype=np.uint16)
            u32 = u16.astype(np.uint32) << 16
            return u32.view(np.float32).copy()

        if st_dtype in _RAW_READ:
            with open(shard_path, "rb") as fh:
                fh.seek(data_start + data_offsets[0])
                raw = fh.read(data_offsets[1] - data_offsets[0])
            return np.frombuffer(raw, dtype=np.uint8).astype(np.float32)

    except Exception:
        pass
    return None


def _parse_header(path: Path):
    with open(path, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hdr_len).rstrip(b" \x00"))
    return {k: v for k, v in hdr.items() if k != "__metadata__"}, 8 + hdr_len


class LayerBreakdownPlugin(BasePlugin):
    name        = "Layer Breakdown"
    version     = "0.1.0"
    description = "Per-tensor statistics: mean, std, sparsity, unique counts, and a sparsity bar chart."

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("🧅 Layer Breakdown"):
            gr.Markdown("### Per-Tensor Weight Statistics")
            gr.Markdown(
                "Analyzes each tensor individually.  *This re-reads tensors from "
                "disk — may take 10-30 seconds for large models.*"
            )

            sparsity_threshold = gr.Slider(
                minimum=1e-6, maximum=1e-2, value=1e-4, step=1e-6,
                label="Sparsity threshold  |v| ≤ ε",
            )
            run_btn     = gr.Button("🔬 Analyze Layers", variant="primary")
            status_box  = gr.Textbox(label="Status", interactive=False)
            stats_table = gr.Dataframe(
                headers=["Tensor", "Shard", "Dtype", "Params", "Mean", "Std",
                         "Min", "Max", "Sparsity %", "Unique Values"],
                datatype=["str","str","str","number","number","number",
                          "number","number","number","number"],
                label="Layer Statistics",
            )
            sparsity_plot = gr.Plot(label="Sparsity by Layer")
            unique_plot   = gr.Plot(label="Unique Values by Layer")

            run_btn.click(
                fn=self._analyze,
                inputs=[self.state["current_metadata"], sparsity_threshold],
                outputs=[status_box, stats_table, sparsity_plot, unique_plot],
            )

    def _analyze(self, metadata: Optional[Dict], threshold: float):
        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        if metadata is None:
            return "❌ No model loaded", pd.DataFrame(), empty_fig, empty_fig

        shard_paths = [Path(p) for p in metadata.get("shard_paths", [metadata.get("file_path", "")])]
        if not shard_paths or not shard_paths[0].exists():
            return "❌ Shard files not accessible from cache", pd.DataFrame(), empty_fig, empty_fig

        # Build shard → header map
        shard_headers = {}
        for sp in shard_paths:
            try:
                hdr, ds = _parse_header(sp)
                shard_headers[sp] = (hdr, ds)
            except Exception:
                pass

        rows: List[Dict] = []

        for sp, (hdr, data_start) in shard_headers.items():
            for tname, info in hdr.items():
                st_dtype = info.get("dtype", "?")
                shape    = info.get("shape", [])
                offsets  = info.get("data_offsets", [0, 0])
                n_params = int(np.prod(shape)) if shape else 0

                fvals = _read_tensor_as_f32(sp, tname, st_dtype, offsets, data_start)
                if fvals is None or len(fvals) == 0:
                    rows.append({
                        "Tensor": tname, "Shard": sp.name, "Dtype": st_dtype,
                        "Params": n_params, "Mean": None, "Std": None,
                        "Min": None, "Max": None, "Sparsity %": None,
                        "Unique Values": None,
                    })
                    continue

                sparsity = float(np.sum(np.abs(fvals) <= threshold) / len(fvals) * 100)
                rows.append({
                    "Tensor":        tname,
                    "Shard":         sp.name,
                    "Dtype":         st_dtype,
                    "Params":        n_params,
                    "Mean":          round(float(np.mean(fvals)), 6),
                    "Std":           round(float(np.std(fvals)),  6),
                    "Min":           round(float(fvals.min()),    6),
                    "Max":           round(float(fvals.max()),    6),
                    "Sparsity %":    round(sparsity, 3),
                    "Unique Values": int(len(np.unique(fvals))),
                })

        if not rows:
            return "❌ No tensors could be read", pd.DataFrame(), empty_fig, empty_fig

        df = pd.DataFrame(rows).sort_values("Sparsity %", ascending=False)

        # Sparsity bar chart
        top = df.dropna(subset=["Sparsity %"]).head(40)
        sp_fig = px.bar(
            top, x="Tensor", y="Sparsity %",
            color="Sparsity %", color_continuous_scale="RdYlGn_r",
            title=f"Sparsity by Layer  (|v| ≤ {threshold:.1e}, top 40)",
        )
        sp_fig.update_layout(xaxis_tickangle=-45, height=450, template="plotly_white")

        # Unique values bar chart
        uq_fig = px.bar(
            top.sort_values("Unique Values"), x="Tensor", y="Unique Values",
            color="Unique Values", color_continuous_scale="Blues",
            title="Unique Values by Layer (top 40 by sparsity)",
        )
        uq_fig.update_layout(xaxis_tickangle=-45, height=450, template="plotly_white")

        status = f"✅ Analyzed {len(rows)} tensors across {len(shard_headers)} shard(s)"
        return status, df, sp_fig, uq_fig
