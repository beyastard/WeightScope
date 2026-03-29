"""
WeightScope Plugin - Lookup Table Compression Estimator
=======================================================

When a model has very few unique weight values (e.g. MiniCPM4's 6,820 unique
values across 433M parameters), it can be stored as an index array plus a
small lookup table rather than storing every value in full.

This plugin calculates:
  - The minimum index bit-width needed to address all unique values
  - Compressed model size for each feasible index width (8, 12, 13, 16 bit)
  - Compression ratio vs the raw on-disk file size(s)
  - Break-even point: at how many unique values does the approach stop saving space
  - A bar chart comparing storage formats

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

import math
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from weightscope.plugins.base import BasePlugin


def _raw_size_bytes(metadata: Dict) -> int:
    """Sum of actual on-disk shard file sizes."""
    total = 0
    for p in metadata.get("shard_paths", [metadata.get("file_path", "")]):
        try:
            total += Path(p).stat().st_size
        except Exception:
            pass
    return total


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


class LookupTableEstimatorPlugin(BasePlugin):
    name        = "LUT Compression"
    version     = "0.1.0"
    description = "Estimates compressed model size using index array + lookup table encoding."
    
    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("📦 LUT Compression"):
            gr.Markdown("### Lookup Table Compression Estimator")
            gr.Markdown(
                "Models with few unique weight values can be stored as a compact "
                "index array pointing into a small lookup table (LUT).  This tab "
                "calculates how much space that saves at each feasible index width."
            )
            
            run_btn      = gr.Button("📐 Calculate", variant="primary")
            summary_md   = gr.Markdown("*Load a model and click Calculate.*")
            results_json = gr.JSON(label="Compression Details")
            size_plot    = gr.Plot(label="Storage Size Comparison")
            ratio_plot   = gr.Plot(label="Compression Ratio by Index Width")
            
            run_btn.click(
                fn=self._calculate,
                inputs=[self.state["current_df"], self.state["current_metadata"]],
                outputs=[summary_md, results_json, size_plot, ratio_plot],
            )    
    
    def _calculate(self, df: Optional[pd.DataFrame], metadata: Optional[Dict]):
        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        if df is None or metadata is None:
            return "*❌ No model loaded.*", {}, empty_fig, empty_fig
        
        unique_count = len(df)
        total_params = int(df["count"].sum())
        raw_bytes    = _raw_size_bytes(metadata)
        dtypes_found = metadata.get("dtypes_found", ["float32"])
        
        # Bits per element in the raw format
        dtype_bits = {
            "float32": 32, "float16": 16, "bfloat16": 16,
            "int8": 8, "uint8": 8, "float8_e4m3fn": 8, "float8_e5m2": 8,
            "int4": 4,
        }
        primary_dtype = dtypes_found[0] if dtypes_found else "float32"
        raw_bpe = dtype_bits.get(primary_dtype, 32)   # bits per element
        
        # Minimum index bits needed
        min_bits = max(1, math.ceil(math.log2(unique_count + 1))) if unique_count > 0 else 1

        # LUT table overhead: unique_count × 32 bits (store values as float32)
        lut_table_bytes = unique_count * 4

        # Evaluate feasible index widths
        candidate_widths = [w for w in (4, 8, 12, 13, 14, 16) if w >= min_bits]
        if not candidate_widths:
            candidate_widths = [min_bits]
        
        formats: List[Dict] = []
        for width in candidate_widths:
            index_bytes  = math.ceil(total_params * width / 8)
            total_lut_bytes = index_bytes + lut_table_bytes
            ratio = raw_bytes / total_lut_bytes if total_lut_bytes > 0 else 1.0
            saving_pct = (1 - total_lut_bytes / raw_bytes) * 100 if raw_bytes > 0 else 0
            formats.append({
                "index_bits":       width,
                "index_array_size": _format_bytes(index_bytes),
                "lut_table_size":   _format_bytes(lut_table_bytes),
                "total_size":       _format_bytes(total_lut_bytes),
                "total_bytes":      total_lut_bytes,
                "compression_ratio": round(ratio, 3),
                "space_saving_pct":  round(saving_pct, 1),
                "feasible":         width >= min_bits,
            })

        # Break-even: how many unique values fills one bit-width to capacity
        break_even = {f"{w}-bit": 2**w for w in (8, 12, 13, 16)}

        result = {
            "unique_values":    unique_count,
            "total_parameters": total_params,
            "raw_file_size":    _format_bytes(raw_bytes),
            "raw_dtype":        primary_dtype,
            "raw_bpe":          raw_bpe,
            "min_index_bits":   min_bits,
            "lut_table_bytes":  _format_bytes(lut_table_bytes),
            "formats":          formats,
            "break_even_unique_values": break_even,
        }

        # Summary markdown
        best = min(formats, key=lambda x: x["total_bytes"])
        summary = (
            f"**{unique_count:,} unique values** across **{total_params:,} parameters**  \n"
            f"Minimum index width needed: **{min_bits} bits**  \n"
            f"Raw file size: **{_format_bytes(raw_bytes)}** ({primary_dtype})  \n"
            f"Best LUT encoding: **{best['index_bits']}-bit index** → "
            f"**{best['total_size']}** ({best['space_saving_pct']:.1f}% smaller)  \n"
            f"LUT table overhead: **{_format_bytes(lut_table_bytes)}** "
            f"({unique_count:,} x 4 bytes float32)"
        )

        # Size bar chart
        labels = [f"Raw ({primary_dtype})"] + [f"{f['index_bits']}-bit LUT" for f in formats]
        sizes  = [raw_bytes] + [f["total_bytes"] for f in formats]
        colors = ["#e74c3c"] + ["#2ecc71" if f["total_bytes"] < raw_bytes else "#e67e22"
                                for f in formats]
        
        size_fig = go.Figure(go.Bar(
            x=labels, y=[s / (1024**3) for s in sizes],
            marker_color=colors,
            text=[f"{s/(1024**3):.3f} GB" for s in sizes],
            textposition="auto",
        ))
        
        size_fig.update_layout(
            title="Storage Size Comparison (GB)",
            yaxis_title="Size (GB)", template="plotly_white", height=400,
        )

        # Ratio line chart
        ratio_fig = go.Figure(go.Scatter(
            x=[f["index_bits"] for f in formats],
            y=[f["compression_ratio"] for f in formats],
            mode="lines+markers+text",
            text=[f"{f['compression_ratio']}x" for f in formats],
            textposition="top center",
            marker=dict(size=10, color="#3498db"),
            line=dict(width=2),
        ))
        
        ratio_fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even (no saving)")
        ratio_fig.update_layout(
            title="Compression Ratio by Index Width",
            xaxis_title="Index Width (bits)", yaxis_title="Compression Ratio (x)",
            template="plotly_white", height=400,
        )

        return summary, result, size_fig, ratio_fig
