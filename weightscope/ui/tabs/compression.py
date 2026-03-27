"""
WeightScope UI - Compression tab

Covers:
  • Uniform quantization simulation (any bit-width 4-16, incl. INT8/INT4)
  • Low-count unique-value removal simulation

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

from ...core.analyzer import WeightAnalyzer


def create_compression_tab():
    """
    Build the Compression analysis tab.

    Returns all components and callbacks needed by ``app_builder``.
    """
    with gr.Tab("🗜️ Compression"):
        gr.Markdown("### Bit-Width Reduction & Unique-Value Analysis")
        compression_md = gr.Markdown("*Load a model to see compression options*")

        gr.Markdown("#### Quantization Simulation")
        gr.Markdown(
            "Simulates uniform linear quantization. "
            "Supports any bit-width from 4 (INT4) to 16 (FP16/INT16)."
        )

        with gr.Row():
            quant_bits   = gr.Slider(minimum=4, maximum=16, value=8, step=1,
                                     label="Target Bit-Width")
            quant_method = gr.Dropdown(choices=["mse", "mae"], value="mse",
                                       label="Error Metric")

        quant_btn     = gr.Button("🧮 Simulate Quantization", variant="secondary")
        quant_results = gr.JSON(label="Quantization Results")

        gr.Markdown("---")
        gr.Markdown("#### Low-Count Unique-Value Removal")
        gr.Markdown(
            "Shows how many parameters are affected if you discard "
            "weight values that occur very rarely."
        )

        low_count_slider  = gr.Slider(minimum=1, maximum=10, value=4, step=1,
                                      label="Remove Values with Count ≤")
        low_count_btn     = gr.Button("🗑️ Simulate Removal", variant="secondary")
        low_count_results = gr.JSON(label="Removal Impact")

    # ─── Callbacks ────────────────────────────────────────────────────────────

    def simulate_quant(df, bits, method):
        if df is None:
            return {"error": "No data loaded"}
        a = WeightAnalyzer()
        a.df = df
        return a.simulate_quantization(int(bits), method)

    def simulate_removal(df, max_count):
        if df is None:
            return {"error": "No data loaded"}
        a = WeightAnalyzer()
        a.df = df
        return a.simulate_low_count_removal(int(max_count))

    return (
        compression_md,
        quant_bits, quant_method, quant_btn, quant_results,
        low_count_slider, low_count_btn, low_count_results,
        simulate_quant, simulate_removal,
    )
