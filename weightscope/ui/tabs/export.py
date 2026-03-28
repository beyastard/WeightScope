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

from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

from ...core import SessionCache
from ...config import OUTPUT_DIR
from ..plotting import (
    create_histogram,
    create_scatter,
    save_figure,
)


def create_export_tab(cache: SessionCache):
    """
    Build the Export tab.

    Returns all components and callbacks needed by ``app_builder``.
    The plot-save callback is returned separately so ``app_builder`` can
    pass the live histogram/scatter figures as inputs.
    """
    with gr.Tab("💾 Export"):
        gr.Markdown("### Export Analysis Data & Plots")
        
        # ── Data export ───────────────────────────────────────────────────────
        gr.Markdown("#### Analysis Data")
        with gr.Row():
            export_format = gr.Dropdown(
                choices=["parquet", "csv", "json"],
                value="parquet",
                label="Data Format",
                info="Parquet recommended for large datasets",
            )

        output_dir = gr.Textbox(
            label="Output Directory",
            placeholder="./output  or  C:/Analysis/Output/",
            value=str(OUTPUT_DIR),
        )
        export_btn    = gr.Button("📤 Export Data", variant="primary")
        export_status = gr.Textbox(label="Export Status", interactive=False)

        gr.Markdown("---")
        
        # ── Plot export ───────────────────────────────────────────────────────
        gr.Markdown("#### Save Plots")
        gr.Markdown(
            "Regenerates and saves all distribution plots using the current "
            "filter settings.  Requires the **Distribution** tab sliders to "
            "be set to the desired range before saving."
        )

        with gr.Row():
            plot_format = gr.Dropdown(
                choices=["png", "svg", "html"],
                value="png",
                label="Plot Format",
                info="PNG for reports · SVG for vector editing · HTML for interactive",
            )
            plot_prefix = gr.Textbox(
                label="Filename Prefix (optional)",
                placeholder="my_model",
                value="",
            )

        with gr.Row():
            vmin_export = gr.Number(value=-2.0,  label="Histogram Value Min")
            vmax_export = gr.Number(value=2.0,   label="Histogram Value Max")
            log_export  = gr.Checkbox(value=True, label="Log Scale (Counts)")

        with gr.Row():
            singletons_export = gr.Checkbox(label="Scatter: Singletons only")
            outliers_export   = gr.Checkbox(label="Scatter: Outliers only")

        save_plots_btn    = gr.Button("🖼️ Save Plots", variant="secondary")
        save_plots_status = gr.Textbox(label="Plot Save Status", interactive=False)

    # ─── Callback ─────────────────────────────────────────────────────────────

    # ── Data export callback ──────────────────────────────────────────────────

    def export_data(model_id, out_dir, fmt):
        if not model_id:
            return "❌ No model loaded"
        return cache.export_data(model_id, out_dir, fmt)
    
    # ── Plot save callback ────────────────────────────────────────────────────

    def save_plots(df: Optional[pd.DataFrame], model_id, out_dir: str,
                   fmt: str, prefix: str,
                   vmin: float, vmax: float, log_y: bool,
                   singletons: bool, outliers: bool) -> str:
        if df is None:
            return "❌ No model loaded — load and analyze a model first"

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        safe_id  = (prefix.strip() or str(model_id or "model")).replace("/", "--")
        saved    = []
        errors   = []

        # Histogram
        try:
            fig_hist = create_histogram(df, float(vmin), float(vmax), bool(log_y))
            dest = out_path / f"{safe_id}_histogram.{fmt}"
            save_figure(fig_hist, dest, fmt)
            saved.append(f"histogram → {dest.name}")
        except Exception as exc:
            errors.append(f"histogram: {exc}")

        # Scatter
        try:
            fig_scatter = create_scatter(df, bool(singletons), bool(outliers))
            dest = out_path / f"{safe_id}_scatter.{fmt}"
            save_figure(fig_scatter, dest, fmt)
            saved.append(f"scatter   → {dest.name}")
        except Exception as exc:
            errors.append(f"scatter: {exc}")

        lines = []
        if saved:
            lines.append(f"✅ Saved {len(saved)} plot(s) to {out_path}/")
            lines.extend(f"   {s}" for s in saved)
        if errors:
            lines.append(f"⚠️  {len(errors)} error(s):")
            lines.extend(f"   {e}" for e in errors)
        return "\n".join(lines)

    return (
        export_format, plot_format, plot_prefix,
        output_dir,
        export_btn, export_status, export_data,
        vmin_export, vmax_export, log_export,
        singletons_export, outliers_export,
        save_plots_btn, save_plots_status, save_plots,
    )
