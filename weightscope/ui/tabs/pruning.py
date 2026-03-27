"""
WeightScope UI - Pruning tab

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

from ...config        import DEFAULT_PRUNING_THRESHOLD
from ...core.analyzer import WeightAnalyzer
from ...utils         import format_number


def create_pruning_tab():
    """
    Build the Pruning candidate analysis tab.

    Returns all components and the analysis callback.
    """
    with gr.Tab("✂️ Pruning"):
        gr.Markdown("### Pruning Candidate Analysis")
        gr.Markdown(
            "Identifies weight values whose absolute magnitude falls below "
            "the chosen threshold ε — these are safe candidates for zeroing out."
        )

        pruning_threshold = gr.Slider(
            minimum=1e-6, maximum=1e-2,
            value=DEFAULT_PRUNING_THRESHOLD,
            step=1e-6,
            label="Pruning Threshold  ε",
        )
        pruning_md    = gr.Markdown("*Adjust the threshold to see pruning candidates*")
        pruning_table = gr.Dataframe(
            headers=["Bit Pattern", "Value", "Count"],
            datatype=["str", "number", "number"],
            label="Pruning Candidates (top 50)",
        )

    # ─── Callback ─────────────────────────────────────────────────────────────

    def analyze_pruning(df, threshold):
        if df is None:
            return "❌ No model loaded", pd.DataFrame()

        a = WeightAnalyzer()
        a.df = df
        results = a.get_pruning_candidates(threshold)

        summary = f"""
### Pruning Analysis  (ε = {threshold:.2e})
| Metric | Value |
|--------|-------|
| **Prunable Parameters** | {format_number(results['prunable_parameters'])} |
| **Sparsity** | {results['sparsity_pct']:.4f} % |
| **Unique Candidates** | {format_number(results['unique_candidates'])} |
"""
        candidates_df = results["candidates"].head(50)
        cols = [c for c in ["bit_pattern", "value", "count"] if c in candidates_df.columns]
        return summary, candidates_df[cols]

    return pruning_threshold, pruning_md, pruning_table, analyze_pruning
