"""
WeightScope Plugin - Entropy & Information Density
==================================================

Computes Shannon entropy and related information-theoretic metrics from the
weight frequency table.  These metrics quantify how "random" or "compressible"
the weight distribution is:

  Shannon entropy  H = −Σ p(v) · log₂(p(v))
    where p(v) = count(v) / total_parameters

  Maximum entropy  H_max = log₂(unique_values)
    (what entropy would be if all unique values appeared equally often)

  Relative entropy  H / H_max
    (1.0 = maximally uniform; 0.0 = single repeated value)

  Theoretical minimum bits per weight
    = H   (Shannon source coding theorem lower bound)

  Actual bits per weight
    = bits used by the storage dtype (e.g. 16 for BF16, 32 for FP32)

  Redundancy
    = actual_bits - theoretical_minimum

  Compression ceiling
    = redundancy / actual_bits
    maximum lossless compression ratio achievable in principle

Also shows the top-N most frequent values (which dominate the entropy
calculation) and a probability-mass pie chart split into:
  - Top 10 values
  - Next 90 values
  - Remainder

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

from typing import Dict, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from weightscope.plugins.base import BasePlugin


_DTYPE_BITS = {
    "float32": 32, "float16": 16, "bfloat16": 16,
    "int8": 8, "uint8": 8, "float8_e4m3fn": 8, "float8_e5m2": 8, "int4": 4,
}


class EntropyAnalysisPlugin(BasePlugin):
    name        = "Entropy & Info Density"
    version     = "0.1.0"
    description = "Shannon entropy, theoretical minimum bits per weight, and compression ceiling."

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("🔢 Entropy"):
            gr.Markdown("### Entropy & Information Density Analysis")
            gr.Markdown(
                "Shannon entropy measures how much information each weight carries. "
                "Low entropy means the distribution is highly predictable and "
                "theoretically very compressible."
            )

            top_n_slider = gr.Slider(
                minimum=5, maximum=50, value=20, step=1,
                label="Top-N frequent values to display",
            )
            run_btn    = gr.Button("🧮 Compute", variant="primary")
            summary_md = gr.Markdown("*Load a model and click Compute.*")
            metrics_json = gr.JSON(label="Information-Theoretic Metrics")

            with gr.Row():
                entropy_plot = gr.Plot(label="Probability Mass Distribution")
                topn_plot    = gr.Plot(label="Top-N Values by Probability")

            run_btn.click(
                fn=self._compute,
                inputs=[self.state["current_df"],
                        self.state["current_metadata"],
                        top_n_slider],
                outputs=[summary_md, metrics_json, entropy_plot, topn_plot],
            )

    def _compute(self, df: Optional[pd.DataFrame], metadata: Optional[Dict], top_n: int):
        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        if df is None or metadata is None:
            return "*❌ No model loaded.*", {}, empty_fig, empty_fig

        values = df["value"].values.astype(np.float64)
        counts = df["count"].values.astype(np.float64)
        total  = counts.sum()
        probs  = counts / total

        unique_count = len(df)

        # Shannon entropy in bits
        H     = float(-np.sum(probs * np.log2(probs + 1e-300)))
        H_max = float(np.log2(unique_count)) if unique_count > 1 else 1.0
        H_rel = H / H_max if H_max > 0 else 1.0

        # Actual bits per weight from dtype
        dtypes_found = metadata.get("dtypes_found", ["float32"])
        primary_dtype = dtypes_found[0] if dtypes_found else "float32"
        actual_bits  = _DTYPE_BITS.get(primary_dtype, 32)

        redundancy          = actual_bits - H
        compression_ceiling = redundancy / actual_bits if actual_bits > 0 else 0.0
        perplexity          = float(2 ** H)

        # Gini coefficient (inequality measure — high = few values dominate)
        sorted_p = np.sort(probs)
        n        = len(sorted_p)
        gini     = float(2 * np.sum(np.arange(1, n+1) * sorted_p) / (n * sorted_p.sum()) - (n+1)/n) if n > 0 else 0.0

        metrics = {
            "total_parameters":      int(total),
            "unique_values":         unique_count,
            "primary_dtype":         primary_dtype,
            "actual_bits_per_weight": actual_bits,
            "shannon_entropy_bits":  round(H, 6),
            "max_entropy_bits":      round(H_max, 6),
            "relative_entropy":      round(H_rel, 6),
            "theoretical_min_bpw":   round(H, 6),
            "redundancy_bits":       round(redundancy, 6),
            "compression_ceiling_pct": round(compression_ceiling * 100, 2),
            "perplexity":            round(perplexity, 2),
            "gini_coefficient":      round(gini, 6),
        }

        # Summary narrative
        compressibility = (
            "extremely high (codebook/weight-sharing model)" if compression_ceiling > 0.7 else
            "high"      if compression_ceiling > 0.5 else
            "moderate"  if compression_ceiling > 0.3 else
            "low (already near information-theoretic limit)"
        )

        summary = f"""
**Shannon Entropy:** {H:.4f} bits  (max possible: {H_max:.4f} bits)  
**Theoretical minimum:** {H:.4f} bits per weight vs **{actual_bits} bits** stored  
**Redundancy:** {redundancy:.4f} bits/weight — compressibility is **{compressibility}**  
**Compression ceiling:** up to **{compression_ceiling*100:.1f}%** lossless reduction is theoretically possible  
**Perplexity:** {perplexity:.1f}  (effective unique values if distribution were uniform)  
**Gini coefficient:** {gini:.4f}  (0 = all values equally likely; 1 = one value dominates)
"""

        # Probability mass pie chart: top 10, next 90, rest
        sorted_df  = df.sort_values("count", ascending=False).reset_index(drop=True)
        mass_top10 = float(sorted_df["count"].iloc[:10].sum() / total) if len(sorted_df) >= 10 else 1.0
        mass_top100 = float(sorted_df["count"].iloc[:100].sum() / total) if len(sorted_df) >= 100 else 1.0
        mass_next90 = mass_top100 - mass_top10
        mass_rest   = max(0.0, 1.0 - mass_top100)

        pie_labels = ["Top 10 values", "Values 11-100", f"Remaining {unique_count-100:,} values"]
        pie_values = [mass_top10, mass_next90, mass_rest]
        if unique_count <= 100:
            pie_labels = ["Top 10 values", f"Remaining {unique_count-10} values"]
            pie_values = [mass_top10, 1.0 - mass_top10]
        if unique_count <= 10:
            pie_labels = [f"{unique_count} total values"]
            pie_values = [1.0]

        pie_fig = go.Figure(go.Pie(
            labels=pie_labels, values=pie_values,
            hole=0.4,
            marker_colors=["#e74c3c", "#f39c12", "#95a5a6"],
        ))
        pie_fig.update_layout(
            title="Probability Mass Concentration",
            template="plotly_white", height=380,
        )

        # Top-N bar chart
        n_show    = min(int(top_n), len(sorted_df))
        top_vals  = sorted_df["value"].values[:n_show]
        top_probs = sorted_df["count"].values[:n_show] / total
        topn_fig  = go.Figure(go.Bar(
            x=[f"{v:.4f}" for v in top_vals],
            y=top_probs,
            marker_color="#3498db",
        ))
        topn_fig.update_layout(
            title=f"Top {n_show} Values by Probability Mass",
            xaxis_title="Weight Value",
            yaxis_title="P(value)",
            xaxis_tickangle=-45,
            template="plotly_white", height=380,
        )

        return summary, metrics, pie_fig, topn_fig
