"""
WeightScope UI - Compare tab

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

import pandas as pd
import gradio as gr

from ..plotting import create_comparison_plot, _empty_fig


def create_compare_tab():
    """
    Build the model comparison tab.

    Returns all components and the comparison callback.
    """
    with gr.Tab("⚖️ Compare"):
        gr.Markdown("### Compare Two Model Weight Distributions")
        gr.Markdown(
            "*Export analysis data from two models (via the Export tab), "
            "then upload both Parquet files here to compare them side-by-side.*"
        )

        with gr.Row():
            compare_file1 = gr.File(label="Model 1  (Parquet / CSV)")
            compare_file2 = gr.File(label="Model 2  (Parquet / CSV)")

        compare_btn  = gr.Button("📊 Compare", variant="secondary")
        compare_plot = gr.Plot(label="Distribution Comparison")

    # ─── Callback ─────────────────────────────────────────────────────────────

    def run_comparison(file1, file2):
        if file1 is None or file2 is None:
            return _empty_fig("Upload two analysis files to compare")

        try:
            df1 = _read_file(file1.name)
            df2 = _read_file(file2.name)
        except Exception as exc:
            return _empty_fig(f"Error reading files: {exc}")

        name1 = _stem(file1.name)
        name2 = _stem(file2.name)
        return create_comparison_plot(df1, df2, name1, name2)

    compare_btn.click(
        fn=run_comparison,
        inputs=[compare_file1, compare_file2],
        outputs=[compare_plot],
    )

    return compare_file1, compare_file2, compare_plot, compare_btn, run_comparison


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _read_file(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _stem(path: str) -> str:
    from pathlib import Path
    return Path(path).stem
