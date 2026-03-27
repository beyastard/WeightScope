"""
WeightScope - Application Builder

Assembles the full Gradio ``Blocks`` UI from individual tab modules,
wires cross-tab event handlers, and mounts registered plugins.

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

from .. import APP_VERSION
from ..core    import ModelLoader, SessionCache
from ..plugins import registry
from ..config  import PLUGINS_DIR

from .tabs.load_model    import create_load_model_tab
from .tabs.overview      import create_overview_tab
from .tabs.distribution  import create_distribution_tab
from .tabs.query         import create_query_tab
from .tabs.compression   import create_compression_tab
from .tabs.pruning       import create_pruning_tab
from .tabs.clip_normalize import create_clip_normalize_tab
from .tabs.compare       import create_compare_tab
from .tabs.export        import create_export_tab


def build_app() -> gr.Blocks:
    """
    Construct and return the fully-wired Gradio application.

    Plugins in ``PLUGINS_DIR`` are auto-discovered and appended as extra tabs.
    """

    # ─── Shared services ──────────────────────────────────────────────────────
    loader = ModelLoader()
    cache  = SessionCache()

    # ─── Discover plugins ─────────────────────────────────────────────────────
    registry.discover(PLUGINS_DIR)

    # ─── Build UI ─────────────────────────────────────────────────────────────
    with gr.Blocks(title=f"WeightScope v{APP_VERSION}", theme=gr.themes.Soft()) as demo:

        gr.Markdown(f"""
# 🔍 WeightScope  v{APP_VERSION}
### Research tool for weight distribution analysis, compression studies, and pruning evaluation

**Supported Models** - any `.safetensors` file: Llama, Mistral, Qwen, Stable Diffusion, Whisper, BERT, and more  
**Supported Dtypes** - FP32 · FP16 · BF16 · FP8 · INT8 · UINT8 · INT4 (packed)  
**Analysis** - bit-exact value counting, compression potential, pruning candidates, clip/normalize simulation
""")

        # ── Load Model tab (returns shared gr.State components) ──────────────
        current_df, current_metadata, current_model_id = create_load_model_tab(loader, cache)

        # ── Overview ─────────────────────────────────────────────────────────
        stats_md, update_overview = create_overview_tab()

        # ── Distribution ─────────────────────────────────────────────────────
        (
            value_min, value_max, log_toggle,
            hist_plot,
            show_singletons, show_outliers,
            scatter_plot,
            on_histogram, on_scatter,
        ) = create_distribution_tab()

        # ── Query ────────────────────────────────────────────────────────────
        (
            v_min, v_max, c_min, c_max,
            search, query_table, query_btn,
            query_preset, custom_inputs,
            run_query,
        ) = create_query_tab()

        # ── Compression ──────────────────────────────────────────────────────
        (
            compression_md,
            quant_bits, quant_method, quant_btn, quant_results,
            low_count_slider, low_count_btn, low_count_results,
            simulate_quant, simulate_removal,
        ) = create_compression_tab()

        # ── Pruning ──────────────────────────────────────────────────────────
        pruning_threshold, pruning_md, pruning_table, analyze_pruning = create_pruning_tab()

        # ── Clip & Normalize ─────────────────────────────────────────────────
        (
            thresh_mode, thresh_val,
            clip_btn, results_json,
            mse_plot, range_plot,
            run_clip_normalize,
        ) = create_clip_normalize_tab()

        # ── Compare ──────────────────────────────────────────────────────────
        compare_file1, compare_file2, compare_plot, compare_btn, _ = create_compare_tab()

        # ── Export ───────────────────────────────────────────────────────────
        (
            export_format, plot_format,
            output_dir, export_btn, export_status,
            export_data,
        ) = create_export_tab(cache)

        # ── Plugin tabs ──────────────────────────────────────────────────────
        shared_state = {
            "current_df":       current_df,
            "current_metadata": current_metadata,
            "current_model_id": current_model_id,
        }
        for plugin in registry.plugins:
            plugin.inject_state(shared_state)
        registry.mount_all(demo)

        # ─── Cross-tab event wiring ───────────────────────────────────────────

        # Overview refreshes whenever metadata changes
        current_metadata.change(
            fn=update_overview,
            inputs=[current_metadata],
            outputs=[stats_md],
        )

        # Distribution histogram
        for trigger in (value_min, value_max, log_toggle):
            trigger.change(
                fn=on_histogram,
                inputs=[current_df, value_min, value_max, log_toggle],
                outputs=[hist_plot],
            )

        # Distribution scatter
        for trigger in (show_singletons, show_outliers):
            trigger.change(
                fn=on_scatter,
                inputs=[current_df, show_singletons, show_outliers],
                outputs=[scatter_plot],
            )

        # Query
        query_btn.click(
            fn=run_query,
            inputs=[current_df, v_min, v_max, c_min, c_max, search, query_preset],
            outputs=[query_table],
        )

        # Compression
        quant_btn.click(
            fn=simulate_quant,
            inputs=[current_df, quant_bits, quant_method],
            outputs=[quant_results],
        )
        low_count_btn.click(
            fn=simulate_removal,
            inputs=[current_df, low_count_slider],
            outputs=[low_count_results],
        )

        # Pruning – live update as threshold slider moves
        pruning_threshold.change(
            fn=analyze_pruning,
            inputs=[current_df, pruning_threshold],
            outputs=[pruning_md, pruning_table],
        )

        # Clip & Normalize
        clip_btn.click(
            fn=run_clip_normalize,
            inputs=[current_df, thresh_mode, thresh_val],
            outputs=[results_json, mse_plot, range_plot],
        )

        # Export
        export_btn.click(
            fn=export_data,
            inputs=[current_model_id, output_dir, export_format],
            outputs=[export_status],
        )

        # Footer
        gr.Markdown(f"""
---
**WeightScope v{APP_VERSION}** · SafeTensors Model Analyzer  
Analysis is cached in `.save_state/` - re-loading the same model is instant  
For large models (>7B params), ensure ≥ 32 GB RAM  
Export data in Parquet format for best downstream performance
""")

    return demo
