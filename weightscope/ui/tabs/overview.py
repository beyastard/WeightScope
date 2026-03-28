"""
WeightScope UI - Overview tab

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

from ...utils import format_number


def create_overview_tab():
    """
    Build the Overview tab.

    Returns
    -------
    stats_md : gr.Markdown
        The component to update when new metadata arrives.
    update_fn : callable
        Function that accepts *metadata* and returns the markdown string.
    """

    def render_stats(metadata):
        if metadata is None:
            return "### Load a model to see statistics"

        total   = metadata.get("total_parameters", 0)
        unique  = metadata.get("unique_patterns",  0)
        tensors = metadata.get("tensor_count",     0)
        dtypes  = ", ".join(metadata.get("dtypes_found", ["unknown"]))
        skipped = metadata.get("skipped_tensors",  [])
        ts      = metadata.get("analysis_timestamp", "N/A")[:19]
        fhash   = metadata.get("file_hash", "N/A")[:16]

        skipped_row = (
            f"\n| **Skipped Tensors** | {len(skipped)} |" if skipped else ""
        )

        return f"""
### 📊 Model Overview
| Metric | Value |
|--------|-------|
| **Total Parameters** | {format_number(total)} |
| **Unique Value Patterns** | {format_number(unique)} |
| **Tensor Count** | {tensors} |
| **Dtypes Analyzed** | `{dtypes}` |
| **Analysis Date** | {ts} |
| **File Hash (first 16)** | `{fhash}…` |{skipped_row}
"""

    with gr.Tab("📊 Overview"):
        stats_md = gr.Markdown("### Load a model to see statistics")

    return stats_md, render_stats
