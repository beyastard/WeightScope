"""
WeightScope Example Plugin - Weight Statistics Summary
======================================================

This plugin demonstrates the full plugin contract.  It adds a new tab that
shows extended descriptive statistics for the loaded model's weights.

To disable this plugin, delete (or rename) this directory.
To use it as a template, copy the directory, rename it, and edit freely.

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

from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd

from weightscope.plugins.base import BasePlugin


class WeightStatisticsPlugin(BasePlugin):
    """Extended descriptive statistics tab."""

    name        = "Weight Statistics"
    version     = "0.1.1"  # plugin version
    description = "Adds an extended descriptive statistics panel for loaded weights."

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("📐 Extended Stats"):
            gr.Markdown("### Extended Weight Statistics")
            gr.Markdown("*Weighted descriptive statistics computed across all unique values.*")

            compute_btn = gr.Button("🧮 Compute Statistics", variant="secondary")
            stats_table = gr.Dataframe(
                headers=["Statistic", "Value"],
                datatype=["str", "str"],
                label="Results",
            )
            hist_plot = gr.Plot(label="Weighted Distribution (percentile buckets)")

            compute_btn.click(
                fn=self._compute,
                inputs=[self.state["current_df"]],
                outputs=[stats_table, hist_plot],
            )

    # ─── Implementation ───────────────────────────────────────────────────────

    def _compute(self,df: Optional[pd.DataFrame],):
        import plotly.graph_objects as go

        if df is None or len(df) == 0:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="No data loaded", showarrow=False)
            return pd.DataFrame(columns=["Statistic", "Value"]), empty_fig

        values = df["value"].values.astype(np.float64)
        counts = df["count"].values.astype(np.float64)
        total  = counts.sum()

        # Weighted statistics
        mean     = np.sum(values * counts) / total
        variance = np.sum((values - mean) ** 2 * counts) / total
        std      = np.sqrt(variance)
        skew     = np.sum((values - mean) ** 3 * counts) / (total * std ** 3) if std else 0.0
        kurt     = np.sum((values - mean) ** 4 * counts) / (total * std ** 4) - 3 if std else 0.0

        # Weighted percentiles via sorted cumsum
        sort_idx = np.argsort(values)
        sv, sc   = values[sort_idx], counts[sort_idx]
        cumw     = np.cumsum(sc)

        def weighted_percentile(p: float) -> float:
            target = total * p / 100
            idx    = np.searchsorted(cumw, target)
            return float(sv[min(idx, len(sv) - 1)])

        rows = [
            ("Count (total params)",   f"{int(total):,}"),
            ("Unique values",           f"{len(df):,}"),
            ("Mean (weighted)",         f"{mean:.6f}"),
            ("Std dev (weighted)",      f"{std:.6f}"),
            ("Skewness (weighted)",     f"{skew:.4f}"),
            ("Excess kurtosis",         f"{kurt:.4f}"),
            ("Min",                     f"{float(values.min()):.6f}"),
            ("P1",                      f"{weighted_percentile(1):.6f}"),
            ("P5",                      f"{weighted_percentile(5):.6f}"),
            ("P25  (Q1)",               f"{weighted_percentile(25):.6f}"),
            ("P50  (median)",           f"{weighted_percentile(50):.6f}"),
            ("P75  (Q3)",               f"{weighted_percentile(75):.6f}"),
            ("P95",                     f"{weighted_percentile(95):.6f}"),
            ("P99",                     f"{weighted_percentile(99):.6f}"),
            ("Max",                     f"{float(values.max()):.6f}"),
            ("IQR  (Q3 − Q1)",          f"{weighted_percentile(75) - weighted_percentile(25):.6f}"),
            ("Sparsity (|v| < 1e-4)",   f"{float(counts[np.abs(values) < 1e-4].sum() / total * 100):.4f} %"),
        ]

        stats_df = pd.DataFrame(rows, columns=["Statistic", "Value"])

        # Percentile histogram
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values  = [weighted_percentile(p) for p in percentiles]
        fig = go.Figure(go.Bar(
            x=[f"P{p}" for p in percentiles],
            y=pct_values,
            marker_color="steelblue",
        ))
        fig.update_layout(
            title="Weighted Percentile Values",
            yaxis_title="Weight Value",
            template="plotly_white",
            height=350,
        )

        return stats_df, fig
