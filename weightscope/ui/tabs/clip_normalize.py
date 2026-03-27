"""
WeightScope UI - Clip & Normalize tab

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

import numpy as np
import gradio as gr
import plotly.graph_objects as go

from ...core.analyzer import WeightAnalyzer
from ..plotting       import _empty_fig


def create_clip_normalize_tab():
    """
    Build the Clip & Normalize simulation tab.

    Returns all components and the simulation callback.
    """
    with gr.Tab("✂️ Clip & Normalize"):
        gr.Markdown("### Clip Outliers + Normalize to  [-1, 1]")
        gr.Markdown(
            "*Simulate the impact of clipping extreme weights and normalizing "
            "the distribution — useful for estimating quantization headroom.*"
        )

        with gr.Row():
            threshold_mode = gr.Dropdown(
                choices=["Absolute", "Standard Deviations (σ)", "Percentile"],
                value="Absolute",
                label="Threshold Mode",
            )
            threshold_value = gr.Slider(
                minimum=0.1, maximum=20.0, value=5.0, step=0.1,
                label="Threshold Value",
                info="|w| < T  |  T × σ  |  T-th percentile",
            )

        with gr.Row():
            gr.Number(value=-1.0, label="Normalize Min", interactive=False)
            gr.Number(value=1.0,  label="Normalize Max", interactive=False)
            gr.Markdown("*Target range is fixed to [-1, 1]*")

        clip_btn = gr.Button("🔬 Simulate Clip + Normalize", variant="primary")

        with gr.Accordion("📊 Results", open=True):
            results_json = gr.JSON(label="Metrics")
            with gr.Row():
                mse_plot   = gr.Plot(label="Error Indicator")
                range_plot = gr.Plot(label="Value Range Comparison")

        gr.Markdown("""
### Interpreting Results
| Metric | Meaning |
|--------|---------|
| **MSE / MAE** | Reconstruction error - lower = less information loss |
| **SNR (dB)** | Signal-to-noise ratio - higher = better preservation |
| **Clipped %** | Fraction of parameters clipped |
| **Bits Saved** | Theoretical bit-width reduction from reduced dynamic range |
| **Unique Reduction** | Fewer unique values → better lossless compression potential |
""")

    # ─── Callback ─────────────────────────────────────────────────────────────

    def run_clip_normalize(df, mode, threshold_value):
        if df is None or len(df) == 0:
            return {"error": "No data loaded"}, _empty_fig("No data"), _empty_fig("No data")

        values = df["value"].values
        counts = df["count"].values

        # Resolve threshold from mode
        if mode == "Absolute":
            threshold = float(threshold_value)
        elif mode == "Standard Deviations (σ)":
            mean      = float(np.sum(values * counts) / np.sum(counts))
            variance  = float(np.sum((values - mean) ** 2 * counts) / np.sum(counts))
            std       = float(np.sqrt(variance))
            threshold = float(threshold_value) * std
        elif mode == "Percentile":
            sorted_idx    = np.argsort(values)
            sorted_vals   = values[sorted_idx]
            sorted_counts = counts[sorted_idx]
            cumsum        = np.cumsum(sorted_counts)
            target        = np.sum(counts) * float(threshold_value) / 100
            idx           = np.searchsorted(cumsum, target)
            threshold     = abs(float(sorted_vals[min(idx, len(sorted_vals) - 1)]))
        else:
            threshold = 5.0

        a    = WeightAnalyzer()
        a.df = df
        res  = a.simulate_clipping_normalization(threshold)

        # MSE indicator
        mse_fig = go.Figure()
        mse_fig.add_trace(go.Indicator(
            mode="number",
            value=res["mse"],
            title={"text": "MSE (Reconstruction Error)"},
            number={"valueformat": ".3e"},
        ))
        mse_fig.update_layout(height=280)

        # Range comparison bar
        orig_rng    = res["original_range"]
        clip_rng    = res["clipped_range"]
        norm_rng    = res["normalized_range"]
        range_fig   = go.Figure(go.Bar(
            x=["Original", "Clipped", "Normalized"],
            y=[
                orig_rng[1] - orig_rng[0],
                clip_rng[1] - clip_rng[0],
                norm_rng[1] - norm_rng[0],
            ],
            marker_color=["steelblue", "orange", "seagreen"],
        ))
        range_fig.update_layout(
            title="Dynamic Range Comparison",
            yaxis_title="Range Width",
            height=280,
        )

        return res, mse_fig, range_fig

    return (
        threshold_mode, threshold_value,
        clip_btn, results_json,
        mse_plot, range_plot,
        run_clip_normalize,
    )
