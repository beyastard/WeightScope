"""
WeightScope UI - Export tab

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

from ...core import SessionCache
from ...config import OUTPUT_DIR


def create_export_tab(cache: SessionCache):
    """
    Build the Export tab.

    Returns all components and the export callback.
    """
    with gr.Tab("💾 Export"):
        gr.Markdown("### Export Analysis Data")

        with gr.Row():
            export_format = gr.Dropdown(
                choices=["parquet", "csv", "json"],
                value="parquet",
                label="Data Format",
                info="Parquet recommended for large datasets",
            )
            plot_format = gr.Dropdown(
                choices=["png", "svg", "html"],
                value="png",
                label="Plot Format (reserved for future use)",
            )

        output_dir = gr.Textbox(
            label="Output Directory",
            placeholder="./output  or  C:/Analysis/Output/",
            value=str(OUTPUT_DIR),
        )
        export_btn    = gr.Button("📤 Export Data", variant="primary")
        export_status = gr.Textbox(label="Export Status", interactive=False)

    # ─── Callback ─────────────────────────────────────────────────────────────

    def export_data(model_id, out_dir, fmt):
        if not model_id:
            return "❌ No model loaded"
        return cache.export_data(model_id, out_dir, fmt)

    return export_format, plot_format, output_dir, export_btn, export_status, export_data
