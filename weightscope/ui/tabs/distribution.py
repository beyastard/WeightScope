"""
WeightScope UI - Distribution tab

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

import gradio as gr

from ..plotting import create_histogram, create_scatter, _empty_fig


def create_distribution_tab():
    """
    Build the Distribution tab with histogram and scatter controls.

    Returns all interactive components so ``app_builder`` can wire events.
    """
    with gr.Tab("📈 Distribution"):
        gr.Markdown("### Weight Distribution Analysis")

        with gr.Row():
            value_min  = gr.Slider(minimum=-20, maximum=20, value=-2,  step=0.1, label="Value Min")
            value_max  = gr.Slider(minimum=-20, maximum=20, value=2,   step=0.1, label="Value Max")
            log_toggle = gr.Checkbox(value=True, label="Log Scale (Counts)")

        hist_plot = gr.Plot(label="Histogram")

        gr.Markdown("---")

        with gr.Row():
            show_singletons = gr.Checkbox(label="Show Only Singletons (count = 1)")
            show_outliers   = gr.Checkbox(label="Show Statistical Outliers")

        scatter_plot = gr.Plot(label="Scatter  (value vs. count)")

    # Internal callbacks -------------------------------------------------------

    def on_histogram(df, vmin, vmax, log):
        if df is None or len(df) == 0:
            return _empty_fig("No data loaded")
        vmin, vmax = (vmin, vmax) if vmin <= vmax else (vmax, vmin)
        return create_histogram(df, vmin, vmax, log)

    def on_scatter(df, singletons, outliers):
        if df is None:
            return _empty_fig("No data loaded")
        return create_scatter(df, singletons, outliers)

    return (
        value_min, value_max, log_toggle,
        hist_plot,
        show_singletons, show_outliers,
        scatter_plot,
        on_histogram, on_scatter,
    )
