"""
WeightScope UI - Load Model tab

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

from ...core  import ModelLoader, WeightAnalyzer, SessionCache
from ...utils import compute_file_hash


def create_load_model_tab(loader: ModelLoader, cache: SessionCache):
    """
    Build the 'Load Model' tab and return shared Gradio State components.

    Returns
    -------
    current_df, current_metadata, current_model_id : gr.State
    """

    def load_model(source: str, local_path: str, hf_model_id: str, progress=gr.Progress()):
        if source == "Local":
            success, message, mem = loader.load_local_model(local_path)
        else:
            hf_model_id = hf_model_id.strip()
            if not hf_model_id:
                return "❌ Please enter a HuggingFace model ID", "", None, None, None, None
            success, message, mem = loader.load_remote_model(hf_model_id)

        if not success:
            return message, "", None, None, None, None

        model_id  = loader.current_model_id
        file_hash = compute_file_hash(loader.current_model_path)

        if cache.check_cache(model_id, file_hash):
            df, metadata = cache.load_state(model_id)
            return (
                message + "\n✅ Loaded from cache",
                f"Loaded: {model_id}",
                df, metadata, mem, model_id,
            )

        analyzer = WeightAnalyzer()
        ok, analysis_msg = analyzer.analyze_model(loader.current_model_path, progress)
        if not ok:
            return analysis_msg, "", None, None, None, None

        cache.save_state(model_id, analyzer.df, analyzer.model_metadata)
        return (
            message + f"\n{analysis_msg}",
            f"Loaded: {model_id}",
            analyzer.df, analyzer.model_metadata, mem, model_id,
        )

    with gr.Tab("📂 Load Model"):
        gr.Markdown("### Select Model Source")

        source_radio = gr.Radio(choices=["Local", "HuggingFace"], value="Local", label="Source")

        with gr.Group():
            local_path = gr.Textbox(
                label="Local Model Directory",
                placeholder="C:/models/my-model/  or  /home/user/models/my-model/",
            )
            gr.Markdown("*Directory must contain `model.safetensors` (and optionally `config.json`)*")

        with gr.Group():
            hf_model_id = gr.Textbox(
                label="HuggingFace Model ID",
                placeholder="amd/AMD-Llama-135m",
            )

        load_btn           = gr.Button("🚀 Load & Analyze", variant="primary", size="lg")
        load_status        = gr.Textbox(label="Status", interactive=False, lines=3)
        model_info_display = gr.Textbox(label="Model Info", interactive=False)
        mem_estimate_json  = gr.JSON(label="Memory Estimate")

        current_df       = gr.State(None)
        current_metadata = gr.State(None)
        current_model_id = gr.State(None)

        load_btn.click(
            fn=load_model,
            inputs=[source_radio, local_path, hf_model_id],
            outputs=[
                load_status, model_info_display,
                current_df, current_metadata,
                mem_estimate_json, current_model_id,
            ],
        )

    return current_df, current_metadata, current_model_id
