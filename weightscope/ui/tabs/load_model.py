"""
WeightScope UI - Load Model tab

Local path entry uses the OS native folder picker dialog (via tkinter) so
users can browse any drive or directory without typing paths by hand.

Folder picker behaviour
-----------------------
Clicking "📁 Browse…" opens the operating system's native folder selection
dialog.  The selected path is written into the "Local Model Directory"
textbox automatically.  The dialog runs in a background thread so the
Gradio event loop is never blocked.
 
On Windows  - uses the built-in Windows Explorer folder dialog.
On macOS    - uses the macOS Finder sheet.
On Linux    - uses the GTK/Qt dialog provided by the desktop environment.
 
If tkinter is not available (some minimal Python installs omit it), the
Browse button is hidden and a note tells the user to type the path manually.
Tkinter ships with the standard CPython installer on all three platforms.
 
The "Local Model Directory" textbox remains fully editable at all times —
manual path entry and paste always work regardless of tkinter availability.

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

import hashlib
import threading
from pathlib import Path
from typing import Optional, Union

import gradio as gr

from ...core  import ModelLoader, WeightAnalyzer, SessionCache
from ...utils import compute_file_hash

# Tkinter availability check
try:
    import tkinter as _tk
    from tkinter import filedialog as _filedialog
    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False


# Native folder picker
def _pick_folder() -> str:
    """
    Open the OS native folder picker dialog and return the selected path.
 
    Runs tkinter in a daemon thread so Gradio's async event loop is not
    blocked while the user navigates the dialog.  Returns an empty string
    if the user cancels or tkinter is unavailable.
    """
    if not _TKINTER_AVAILABLE:
        return ""
 
    result: list[str] = []
 
    def _run() -> None:
        try:
            root = _tk.Tk()
            root.withdraw()                       # hide the blank root window
            root.wm_attributes("-topmost", True)  # dialog floats above browser
            path = _filedialog.askdirectory(
                title="Select Model Directory",
                mustexist=True,
            )
            root.destroy()
            result.append(path or "")
        except Exception:
            result.append("")
 
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=300)   # wait up to 5 minutes for user to choose (too much?)
    return result[0] if result else ""


# Cache hash helper
def _combined_hash(paths) -> str:
    """
    SHA-256 chain across all shard files (order-stable).
    Any change to any shard invalidates the cache.
    """
    h = hashlib.sha256()
    for p in paths:
        h.update(compute_file_hash(p).encode())
    return h.hexdigest()


def create_load_model_tab(loader: ModelLoader, cache: SessionCache):
    """
    Build the 'Load Model' tab and return shared Gradio State components.
    
    Parameters
    ----------
    loader       : ModelLoader instance (shared across the app).
    cache        : SessionCache instance (shared across the app).
    browse_root  : Root directory shown in the folder browser.
                   Defaults to WEIGHTSCOPE_BROWSE_ROOT env-var → home dir.

    Returns
    -------
    current_df, current_metadata, current_model_id : gr.State
    """
    
    def load_model(source: str, local_path: str, hf_model_id: str, progress=gr.Progress()):
        if source == "Local":
            success, message, mem = loader.load_local_model(local_path.strip())
        else:
            hf_model_id = hf_model_id.strip()
            if not hf_model_id:
                return "❌ Please enter a HuggingFace model ID", "", None, None, None, None
            success, message, mem = loader.load_remote_model(hf_model_id)
 
        if not success:
            return message, "", None, None, None, None
 
        model_id  = loader.current_model_id
        file_hash = _combined_hash(loader.current_model_paths)
 
        if cache.check_cache(model_id, file_hash):
            df, metadata = cache.load_state(model_id)
            return (
                message + "\n✅ Loaded from cache",
                f"Loaded: {model_id}",
                df, metadata, mem, model_id,
            )
 
        analyzer = WeightAnalyzer()
        ok, analysis_msg = analyzer.analyze_model(loader.current_model_paths, progress)
        if not ok:
            return analysis_msg, "", None, None, None, None
 
        cache.save_state(model_id, analyzer.df, analyzer.model_metadata)
        return (
            message + f"\n{analysis_msg}",
            f"Loaded: {model_id}",
            analyzer.df, analyzer.model_metadata, mem, model_id,
        )
 
    # ── Browse callback ───────────────────────────────────────────────────────
 
    def browse_for_folder(current_path: str) -> str:
        """
        Open the OS native folder picker.
        Returns the chosen path, or the unchanged current_path if cancelled.
        """
        chosen = _pick_folder()
        return chosen if chosen else current_path
 
    # ── UI layout ─────────────────────────────────────────────────────────────
 
    with gr.Tab("📂 Load Model"):
        gr.Markdown("### Select Model Source")
 
        source_radio = gr.Radio(
            choices=["Local", "HuggingFace"],
            value="Local",
            label="Source",
        )
 
        # ── Local section ─────────────────────────────────────────────────────
        with gr.Group() as local_group:
            gr.Markdown("#### Local Model Directory")
 
            with gr.Row():
                local_path = gr.Textbox(
                    label="Path",
                    placeholder=(
                        "C:/AI/Models/my-model  or  D:/models/llama-3  or  "
                        "/home/user/models/my-model"
                    ),
                    scale=5,
                    container=True,
                )
                if _TKINTER_AVAILABLE:
                    browse_btn = gr.Button(
                        "📁 Browse…",
                        variant="secondary",
                        scale=1,
                        min_width=110,
                    )
                else:
                    browse_btn = None
 
            if not _TKINTER_AVAILABLE:
                gr.Markdown(
                    "⚠️ *`tkinter` is not available in this Python environment — "
                    "type or paste the model directory path above.  "
                    "Install the standard CPython distribution to enable the "
                    "Browse button.*"
                )
            else:
                gr.Markdown(
                    "*Click **📁 Browse…** to open the system folder picker, or "
                    "type / paste the path directly.  The directory must contain "
                    "`model.safetensors` or sharded "
                    "`model-00001-of-NNNNN.safetensors` files.*"
                )
 
        # ── HuggingFace section ───────────────────────────────────────────────
        with gr.Group(visible=False) as hf_group:
            gr.Markdown("#### HuggingFace Model ID")
            hf_model_id = gr.Textbox(
                label="Model ID",
                placeholder="amd/AMD-Llama-135m  or  openbmb/MiniCPM4-0.5B",
            )
            gr.Markdown(
                "*Enter the model ID exactly as it appears on "
                "[huggingface.co/models](https://huggingface.co/models).  "
                "All shards are discovered and downloaded automatically.*"
            )
 
        # ── Shared controls ───────────────────────────────────────────────────
        load_btn           = gr.Button("🚀 Load & Analyze", variant="primary", size="lg")
        load_status        = gr.Textbox(label="Status", interactive=False, lines=3)
        model_info_display = gr.Textbox(label="Model Info", interactive=False)
        mem_estimate_json  = gr.JSON(label="Memory Estimate")
 
        current_df       = gr.State(None)
        current_metadata = gr.State(None)
        current_model_id = gr.State(None)
 
        # ── Event wiring ──────────────────────────────────────────────────────
 
        # Source toggle — show Local or HuggingFace group
        def _toggle_source(source: str):
            return (
                gr.update(visible=(source == "Local")),
                gr.update(visible=(source == "HuggingFace")),
            )
 
        source_radio.change(
            fn=_toggle_source,
            inputs=[source_radio],
            outputs=[local_group, hf_group],
        )
 
        # Browse button — opens OS native folder picker
        if browse_btn is not None:
            browse_btn.click(
                fn=browse_for_folder,
                inputs=[local_path],
                outputs=[local_path],
            )
 
        # Load & Analyze
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
