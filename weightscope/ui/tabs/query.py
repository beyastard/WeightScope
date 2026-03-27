"""
WeightScope UI - Query tab

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
import pandas as pd


# ─── Preset definitions ───────────────────────────────────────────────────────

_PRESETS = {
    "Custom":                         dict(vmin=-2,    vmax=2,    cmin=1, cmax=100_000),
    "🌱 Pruning Candidates (|v|<1e-4)": dict(vmin=-1e-4, vmax=1e-4, cmin=1, cmax=1_000_000),
    "🔍 Singletons (count=1)":          dict(vmin=-20,   vmax=20,   cmin=1, cmax=1),
    "📉 Rare Values (count 1-10)":      dict(vmin=-20,   vmax=20,   cmin=1, cmax=10),
    "📈 High-Frequency (count>10,000)": dict(vmin=-20,   vmax=20,   cmin=10_000, cmax=10_000_000),
    "🎯 Near Zero (|v|<1e-3)":          dict(vmin=-1e-3, vmax=1e-3, cmin=1, cmax=1_000_000),
    "⚡ Extreme Values (|v|>5)":        dict(vmin=-20,   vmax=-5,   cmin=1, cmax=1_000_000),
    "🗑️ Low-Count (count≤4)":           dict(vmin=-20,   vmax=20,   cmin=1, cmax=4),
}


def create_query_tab():
    """
    Build the Query tab with preset selector and custom filter inputs.

    Returns all components and callbacks needed by ``app_builder``.
    """
    with gr.Tab("🔎 Query"):
        gr.Markdown("### Filter and Search Weight Values")

        query_preset = gr.Dropdown(
            choices=list(_PRESETS.keys()),
            value="Custom",
            label="Query Preset",
        )

        with gr.Group(visible=True) as custom_inputs:
            with gr.Row():
                v_min = gr.Number(value=-2,       label="Value Min")
                v_max = gr.Number(value=2,        label="Value Max")
                c_min = gr.Number(value=1,        label="Count Min")
                c_max = gr.Number(value=100_000,  label="Count Max")
            search = gr.Textbox(label="Search (bit pattern or value substring)")

        query_btn   = gr.Button("🔍 Run Query", variant="secondary")
        query_table = gr.Dataframe(
            headers=["Bit Pattern", "Value", "Count"],
            datatype=["str", "number", "number"],
            label="Results (top 100)",
        )

    # ─── Callbacks ────────────────────────────────────────────────────────────

    def apply_preset(preset: str):
        p       = _PRESETS.get(preset, _PRESETS["Custom"])
        visible = preset == "Custom"
        return (
            p["vmin"], p["vmax"],
            p["cmin"], p["cmax"],
            "",
            gr.update(visible=visible),
        )

    def run_query(df, vmin, vmax, cmin, cmax, search_term, preset):
        if df is None:
            return pd.DataFrame()

        # For "Extreme Values" we need both tails
        if isinstance(preset, str) and "Extreme" in preset:
            neg = df[(df["value"] <= -5) & (df["count"] >= cmin) & (df["count"] <= cmax)]
            pos = df[(df["value"] >=  5) & (df["count"] >= cmin) & (df["count"] <= cmax)]
            result = pd.concat([neg, pos])
        else:
            result = df[
                (df["value"] >= vmin) & (df["value"] <= vmax) &
                (df["count"] >= cmin) & (df["count"] <= cmax)
            ]

        if search_term:
            mask = (
                result["bit_pattern"].astype(str).str.contains(search_term, case=False, na=False)
                | result["value"].astype(str).str.contains(search_term, na=False)
            )
            result = result[mask]

        cols = [c for c in ["bit_pattern", "value", "count"] if c in result.columns]
        return result.sort_values("count", ascending=False).head(100)[cols]

    # Wire preset → inputs
    query_preset.change(
        fn=apply_preset,
        inputs=[query_preset],
        outputs=[v_min, v_max, c_min, c_max, search, custom_inputs],
    )

    return (
        v_min, v_max, c_min, c_max,
        search, query_table, query_btn,
        query_preset, custom_inputs,
        run_query,
    )
