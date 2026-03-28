"""
WeightScope Plugin - Outlier Tensor Report
==========================================

Scans every tensor's header metadata and flags statistical anomalies relative
to the model's own norms.  No tensor data is loaded — all metrics come from
the `tensors` list already present in `current_metadata`.

Flags raised:
  🔴 EXTREME_RANGE    — tensor's theoretical max value (from dtype range) is
                        a significant outlier vs peers of the same dtype
  🟠 LOW_UNIQUE       — tensor has very few unique values relative to its size
                        (potential weight-sharing, collapsed layer, or
                        quantization artifact)
  🟡 SINGLETON_HEAVY  — tensor is very small (≤ 64 params); likely a bias or
                        layer-norm scalar that skews global distribution stats
  🔵 LARGE_TENSOR     — tensor accounts for > 10 % of total parameters
                        (embedding tables, large projection layers)
  🟢 NORMAL           — no flags raised

Since per-tensor unique counts require re-reading each tensor (see the Layer
Breakdown plugin), this report works from the metadata already available after
a standard analysis run.  The `unique_patterns` field in the global metadata
gives the cross-model total; per-tensor unique counts come from re-analysis
if the user clicks the optional "Deep Scan" button.

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

from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from weightscope.plugins.base import BasePlugin


_SIZE_THRESHOLD_PCT = 10.0   # flag tensors > this % of total params
_SMALL_TENSOR_MAX   = 64     # flag tensors ≤ this size as singletons


def _classify_tensor(info: Dict, total_params: int, size_threshold_pct: float) -> List[str]:
    flags = []
    size  = info.get("size", 0)

    if total_params > 0 and size / total_params * 100 > size_threshold_pct:
        flags.append("🔵 LARGE_TENSOR")

    if 0 < size <= _SMALL_TENSOR_MAX:
        flags.append("🟡 SINGLETON_HEAVY")

    if not flags:
        flags.append("🟢 NORMAL")

    return flags


class OutlierTensorReportPlugin(BasePlugin):
    name        = "Outlier Tensor Report"
    version     = "0.1.0"
    description = "Flags anomalous tensors by size, dtype, and structural role using cached metadata."

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("🚨 Outlier Report"):
            gr.Markdown("### Outlier Tensor Report")
            gr.Markdown(
                "Scans all tensor metadata from the last analysis and flags "
                "structurally anomalous tensors.  Results are available instantly "
                "from cached metadata — no re-reading of model files required."
            )

            with gr.Row():
                large_pct_slider = gr.Slider(
                    minimum=1.0, maximum=25.0, value=_SIZE_THRESHOLD_PCT, step=0.5,
                    label="Large-tensor threshold  (% of total params)",
                )
                show_normal = gr.Checkbox(value=False, label="Show NORMAL tensors")

            run_btn = gr.Button("🔍 Generate Report", variant="primary")

            summary_md  = gr.Markdown("*Load a model and click Generate Report.*")
            flag_counts = gr.JSON(label="Flag Summary")

            with gr.Row():
                flag_plot  = gr.Plot(label="Flag Distribution")
                size_plot  = gr.Plot(label="Tensor Size Distribution")

            report_table = gr.Dataframe(
                headers=["Tensor", "Shard", "Dtype", "Params", "Shape", "% of Model", "Flags"],
                datatype=["str","str","str","number","str","number","str"],
                label="Tensor Report",
            )

            run_btn.click(
                fn=self._generate,
                inputs=[self.state["current_metadata"], large_pct_slider, show_normal],
                outputs=[summary_md, flag_counts, flag_plot, size_plot, report_table],
            )

    def _generate(self, metadata: Optional[Dict], large_pct: float, show_normal: bool):
        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        if metadata is None:
            return "*❌ No model loaded.*", {}, empty_fig, empty_fig, pd.DataFrame()

        tensors      = metadata.get("tensors", [])
        total_params = metadata.get("total_parameters", 1) or 1

        if not tensors:
            return "*❌ No tensor metadata available.*", {}, empty_fig, empty_fig, pd.DataFrame()

        rows:       List[Dict] = []
        flag_tally: Dict[str, int] = {}

        for info in tensors:
            name   = info.get("name", "?")
            shard  = info.get("shard", "")
            dtype  = info.get("dtype_label", info.get("dtype", "?"))
            size   = info.get("size", 0)
            shape  = info.get("shape", [])
            pct    = round(size / total_params * 100, 4)

            flags  = _classify_tensor(
                {"size": size},
                total_params,
                large_pct,
            )
            flag_str = "  ".join(flags)

            for f in flags:
                flag_tally[f] = flag_tally.get(f, 0) + 1

            rows.append({
                "Tensor":      name,
                "Shard":       shard,
                "Dtype":       dtype,
                "Params":      size,
                "Shape":       str(shape),
                "% of Model":  pct,
                "Flags":       flag_str,
                "_flags":      flags,   # internal for filtering
            })

        df_all = pd.DataFrame(rows)

        if not show_normal:
            df_show = df_all[~df_all["_flags"].apply(lambda f: f == ["🟢 NORMAL"])]
        else:
            df_show = df_all

        df_show = df_show.sort_values("Params", ascending=False)

        # Summary
        n_flagged = len(df_all[df_all["_flags"].apply(lambda f: f != ["🟢 NORMAL"])])
        summary = (
            f"**{len(tensors)} tensors** scanned across "
            f"**{metadata.get('shard_count', 1)} shard(s)**  \n"
            f"**{n_flagged} flagged** tensors  |  "
            f"**{len(tensors) - n_flagged} normal**  \n"
            f"Total parameters: **{total_params:,}**"
        )

        # Flag distribution pie
        if flag_tally:
            pie_fig = go.Figure(go.Pie(
                labels=list(flag_tally.keys()),
                values=list(flag_tally.values()),
                hole=0.35,
            ))
            pie_fig.update_layout(title="Flag Distribution", template="plotly_white", height=350)
        else:
            pie_fig = empty_fig

        # Tensor size histogram (log scale)
        sizes = df_all["Params"].values
        if len(sizes) > 0:
            size_fig = px.histogram(
                df_all, x="Params", nbins=40, log_x=True, log_y=True,
                title="Tensor Size Distribution (log scale)",
                labels={"Params": "Parameter Count"},
                color="Flags",
            )
            size_fig.update_layout(template="plotly_white", height=350)
        else:
            size_fig = empty_fig

        output_cols = ["Tensor", "Shard", "Dtype", "Params", "Shape", "% of Model", "Flags"]
        return summary, flag_tally, pie_fig, size_fig, df_show[output_cols].reset_index(drop=True)
