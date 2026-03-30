"""
WeightScope Plugin - Transformer Pruning
=========================================

Removes the least-important attention heads and/or FFN neurons from every
transformer layer, producing a structurally smaller model that is immediately
runnable without any fine-tuning.

Importance scoring (training-free, magnitude-based)
----------------------------------------------------
Full pruning computes importance scores by running the model on calibration
data and summing gradient-weighted activations.  That requires PyTorch and a
forward pass.

This plugin uses **weight-magnitude proxies** instead, which need no forward
pass and no calibration data:

* **Attention head importance** — for each head h in layer l, compute the
  L2 norm of the concatenated Q/K/V/O weight columns corresponding to that
  head.  Heads whose norm falls below the per-layer threshold are pruned.

* **FFN neuron importance** — for each neuron (row of the up/gate projection),
  compute the L2 norm of its weight row.  The same threshold logic applies.

This is a recognized approximation; it works best at modest pruning ratios
(≤ 30 %).  At aggressive ratios (> 40 %) the pruned model may show quality
degradation that can be recovered with a short fine-tuning run.

What gets written to disk
--------------------------
* Modified `model.safetensors` (or shards) with head/neuron rows zeroed out
  (soft pruning — fast, safe, reversible) OR structurally removed rows
  (hard pruning — changes tensor dimensions, requires config update)
* Updated `config.json`  (for hard pruning:  num_attention_heads,
  intermediate_size are updated)
* All tokenizer / config support files are copied unchanged

Output directory
-----------------
``<output_base>/<model_name>-trans_pr``
e.g. ``D:/models/amd--AMD-Llama-135m-trans_pr``

Pruning modes
-------------
* **Soft pruning (zero masking)** — sets pruned head/neuron weights to zero.
  Config unchanged.  Drop-in replacement; effect is immediate.

* **Hard pruning (structural)** — physically removes pruned rows, reducing
  tensor dimensions.  Config is updated.  Produces a genuinely smaller model
  that loads and infers faster.  *GQA / MQA models use hard pruning for
  attention only when num_kv_heads divides evenly into the target head count.*

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
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from weightscope.plugins.base import BasePlugin
from weightscope.utils.pruning_utils import (
    ArchitectureProfile,
    copy_support_files,
    detect_architecture,
    iter_tensors,
    load_config,
    make_output_name,
    resolve_model_dir,
    save_model_dir,
    update_config,
)


# ─── Head / neuron scoring ────────────────────────────────────────────────────

def _score_attention_heads(
    tensors:  Dict[str, np.ndarray],
    profile:  ArchitectureProfile,
    layer_idx: int,
    n_heads:   int,
    head_dim:  int,
) -> np.ndarray:
    """
    Return a float32 array of shape (n_heads,) with the L2 importance score
    for each attention head in *layer_idx*.

    For GQA models, KV heads are repeated to match the query head count for
    scoring purposes (the actual pruning logic handles the asymmetry).
    """
    scores = np.zeros(n_heads, dtype=np.float32)

    def _get(pat: str) -> Optional[np.ndarray]:
        name = pat.replace("{layer}", str(layer_idx))
        return tensors.get(name)

    # Fused QKV (GPT-2, GPT-NeoX, etc.) — single tensor [3*head_dim*n_heads, ...]
    if profile.fused_qkv_pat:
        fused = _get(profile.fused_qkv_pat)
        if fused is not None:
            # Shape: [hidden, 3 * hidden]  or  [3 * hidden, hidden]
            # Normalise to [3 * n_heads * head_dim, hidden]
            if fused.shape[0] == 3 * n_heads * head_dim:
                mat = fused
            else:
                mat = fused.T
            for h in range(n_heads):
                q_slice = mat[h * head_dim              : (h+1) * head_dim]
                k_slice = mat[n_heads*head_dim + h*head_dim : n_heads*head_dim + (h+1)*head_dim]
                v_slice = mat[2*n_heads*head_dim + h*head_dim : 2*n_heads*head_dim + (h+1)*head_dim]
                scores[h] = float(
                    np.linalg.norm(q_slice) +
                    np.linalg.norm(k_slice) +
                    np.linalg.norm(v_slice)
                )
        return scores

    # Separate Q / K / V / O projections
    q = _get(profile.q_proj_pat)
    k = _get(profile.k_proj_pat)
    v = _get(profile.v_proj_pat)
    o = _get(profile.o_proj_pat)

    if q is None:
        return scores   # architecture not supported for this layer

    # Q shape: [n_heads * head_dim, hidden]
    # K/V shape may differ for GQA:  [n_kv_heads * head_dim, hidden]
    # We score by Q head norm + averaged K/V norms
    n_kv_heads  = k.shape[0] // head_dim if k is not None else n_heads
    kv_repeat   = n_heads // n_kv_heads  # repetition factor for GQA

    for h in range(n_heads):
        kv_h  = h // kv_repeat
        q_row = q[h * head_dim : (h+1) * head_dim]
        score = float(np.linalg.norm(q_row))

        if k is not None:
            k_row  = k[kv_h * head_dim : (kv_h+1) * head_dim]
            score += float(np.linalg.norm(k_row))
        if v is not None:
            v_row  = v[kv_h * head_dim : (kv_h+1) * head_dim]
            score += float(np.linalg.norm(v_row))
        if o is not None:
            # O proj: [hidden, n_heads * head_dim] — columns correspond to heads
            o_col  = o[:, h * head_dim : (h+1) * head_dim]
            score += float(np.linalg.norm(o_col))

        scores[h] = score

    return scores


def _score_ffn_neurons(
    tensors:     Dict[str, np.ndarray],
    profile:     ArchitectureProfile,
    layer_idx:   int,
) -> np.ndarray:
    """
    Return a float32 array of shape (intermediate_size,) with the L2 norm of
    each FFN neuron's weight row in the up/gate projection.
    """
    up_name = profile.ffn_up_pat.replace("{layer}", str(layer_idx))
    up = tensors.get(up_name)
    if up is None:
        return np.array([], dtype=np.float32)

    # Shape: [intermediate_size, hidden_size]
    scores = np.linalg.norm(up, axis=1).astype(np.float32)

    if profile.ffn_gate_pat:
        gate_name = profile.ffn_gate_pat.replace("{layer}", str(layer_idx))
        gate = tensors.get(gate_name)
        if gate is not None:
            scores += np.linalg.norm(gate, axis=1).astype(np.float32)

    return scores


# ─── Main plugin ─────────────────────────────────────────────────────────────

class TransformerPruningPlugin(BasePlugin):
    name        = "Transformer Pruning"
    version     = "0.1.0"
    description = (
        "Removes the least-important attention heads and FFN neurons using "
        "weight-magnitude importance scoring (no forward pass required)."
    )

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("🔧 Transformer Pruning"):
            gr.Markdown("## Transformer Pruning")
            gr.Markdown(
                "Removes low-importance attention heads and/or FFN neurons from "
                "every layer using weight-magnitude scoring.  No calibration data "
                "or forward pass is required.  Best results at ≤ 30 % pruning ratio; "
                "recommend fine-tuning after aggressive pruning (> 40 %).\n\n"
                "> **Tip:** Run the **Layer Breakdown** plugin first to identify "
                "which layers are already sparse — those are the safest to prune."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### What to prune")
                    prune_heads = gr.Checkbox(value=True,  label="Prune attention heads")
                    prune_ffn   = gr.Checkbox(value=True,  label="Prune FFN neurons")
                    hard_prune  = gr.Checkbox(value=False,
                        label="Hard pruning (structurally removes rows — smaller model, "
                              "requires config update; soft pruning just zeroes weights)")

                with gr.Column(scale=2):
                    gr.Markdown("### Pruning ratio")
                    head_ratio = gr.Slider(
                        minimum=0.0, maximum=0.5, value=0.1, step=0.01,
                        label="Head pruning ratio  (fraction of heads to remove per layer)",
                        info="0.1 = remove the weakest 10 % of heads per layer",
                    )
                    ffn_ratio = gr.Slider(
                        minimum=0.0, maximum=0.5, value=0.1, step=0.01,
                        label="FFN neuron pruning ratio",
                        info="0.1 = remove the weakest 10 % of FFN neurons per layer",
                    )
                    uniform_layers = gr.Checkbox(
                        value=True,
                        label="Uniform ratio per layer  (uncheck for global ranking)",
                    )

            with gr.Row():
                output_base = gr.Textbox(
                    label="Output Base Directory",
                    placeholder="D:/models",
                    value="output",
                )
                storage_fmt = gr.Dropdown(
                    choices=["F32", "BF16"],
                    value="F32",
                    label="Output weight dtype",
                )

            with gr.Row():
                analyze_btn = gr.Button("🔍 Preview (dry run)", variant="secondary")
                prune_btn   = gr.Button("🔧 Prune & Save",       variant="primary")

            status_box  = gr.Textbox(label="Status", interactive=False, lines=3)
            preview_md  = gr.Markdown("*Click Preview to see which heads/neurons will be pruned.*")

            with gr.Row():
                stats_json  = gr.JSON(label="Pruning Statistics")
                score_plot  = gr.Plot(label="Head Importance Scores (layer 0)")

            analyze_btn.click(
                fn=self._preview,
                inputs=[
                    self.state["current_metadata"],
                    prune_heads, prune_ffn, hard_prune,
                    head_ratio, ffn_ratio, uniform_layers,
                ],
                outputs=[status_box, preview_md, stats_json, score_plot],
            )
            prune_btn.click(
                fn=self._prune,
                inputs=[
                    self.state["current_metadata"],
                    prune_heads, prune_ffn, hard_prune,
                    head_ratio, ffn_ratio, uniform_layers,
                    output_base, storage_fmt,
                ],
                outputs=[status_box, preview_md, stats_json, score_plot],
            )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_model_dir(self, metadata):
        return resolve_model_dir(metadata)

    def _analyze(self, metadata, prune_heads, prune_ffn, head_ratio, ffn_ratio, uniform_layers):
        """
        Load all tensors and compute per-layer importance scores.
        Returns (config, profile, all_tensors, head_scores, ffn_scores, shard_paths).
        """
        model_dir   = self._get_model_dir(metadata)
        config      = load_config(model_dir)
        profile     = detect_architecture(config)
        n_layers    = int(config.get(profile.n_layers_key, 0))
        n_heads     = int(config.get(profile.n_heads_key,  0))
        hidden_size = int(config.get(profile.hidden_size_key, 0))
        # Use explicit head_dim from config if present (Qwen2.5 style),
        # otherwise derive from hidden_size // n_heads
        if "head_dim" in config:
            head_dim = int(config["head_dim"])
        elif n_heads:
            head_dim = hidden_size // n_heads
        else:
            head_dim = 0

        n_kv_heads = int(config.get(profile.n_kv_heads_key, n_heads))

        # Guard: if core dimensions are missing, raise a clear error rather than
        # producing a cryptic ZeroDivisionError deep in the scoring code
        if n_layers == 0:
            raise ValueError(
                f"Could not read num_hidden_layers from config.json "
                f"(key: '{profile.n_layers_key}').  "
                f"config.json may not be in the expected location."
            )
        if n_heads == 0 or head_dim == 0:
            raise ValueError(
                f"Could not determine attention head dimensions: "
                f"num_attention_heads={n_heads}, hidden_size={hidden_size}, "
                f"head_dim={head_dim}.  "
                f"Check that config.json exists alongside the model files."
            )

        shard_paths = [Path(p) for p in metadata.get("shard_paths", [model_dir / "model.safetensors"])]
        all_tensors = {name: arr for name, arr, _ in iter_tensors(shard_paths)}

        head_scores_per_layer: List[np.ndarray] = []
        ffn_scores_per_layer:  List[np.ndarray] = []

        for layer_idx in range(n_layers):
            if prune_heads:
                hs = _score_attention_heads(all_tensors, profile, layer_idx, n_heads, head_dim)
                head_scores_per_layer.append(hs)
            if prune_ffn:
                fs = _score_ffn_neurons(all_tensors, profile, layer_idx)
                ffn_scores_per_layer.append(fs)

        return (config, profile, all_tensors, shard_paths,
                head_scores_per_layer, ffn_scores_per_layer,
                n_layers, n_heads, head_dim, n_kv_heads,
                int(config.get(profile.intermediate_key, 0)))

    def _build_masks(
        self,
        head_scores_per_layer, ffn_scores_per_layer,
        n_heads, head_ratio, ffn_ratio, uniform_layers, n_layers,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Build boolean keep_masks for heads and FFN neurons.
        Returns (head_keep_per_layer, ffn_keep_per_layer).
        """
        head_keep_per_layer: List[np.ndarray] = []
        ffn_keep_per_layer:  List[np.ndarray] = []

        if head_scores_per_layer:
            if uniform_layers:
                for scores in head_scores_per_layer:
                    n_prune = max(0, int(len(scores) * float(head_ratio)))
                    thresh  = np.sort(scores)[n_prune] if n_prune < len(scores) else np.inf
                    head_keep_per_layer.append(scores >= thresh)
            else:
                # Global ranking: pool all scores, find global threshold
                all_scores = np.concatenate(head_scores_per_layer)
                total      = len(all_scores)
                n_prune    = int(total * float(head_ratio))
                thresh     = np.sort(all_scores)[n_prune] if n_prune < total else np.inf
                for scores in head_scores_per_layer:
                    head_keep_per_layer.append(scores >= thresh)

        if ffn_scores_per_layer:
            if uniform_layers:
                for scores in ffn_scores_per_layer:
                    if len(scores) == 0:
                        ffn_keep_per_layer.append(np.array([], dtype=bool))
                        continue
                    n_prune = int(len(scores) * float(ffn_ratio))
                    thresh  = np.sort(scores)[n_prune] if n_prune < len(scores) else np.inf
                    ffn_keep_per_layer.append(scores >= thresh)
            else:
                all_scores = np.concatenate([s for s in ffn_scores_per_layer if len(s)])
                total      = len(all_scores)
                n_prune    = int(total * float(ffn_ratio))
                thresh     = np.sort(all_scores)[n_prune] if n_prune < total else np.inf
                for scores in ffn_scores_per_layer:
                    if len(scores) == 0:
                        ffn_keep_per_layer.append(np.array([], dtype=bool))
                    else:
                        ffn_keep_per_layer.append(scores >= thresh)

        return head_keep_per_layer, ffn_keep_per_layer

    def _preview(self, metadata, prune_heads, prune_ffn, hard_prune,
                 head_ratio, ffn_ratio, uniform_layers):
        import plotly.graph_objects as go

        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        if metadata is None:
            return "❌ No model loaded", "*Load a model first.*", {}, empty_fig

        try:
            (config, profile, all_tensors, shard_paths,
             head_scores, ffn_scores,
             n_layers, n_heads, head_dim, n_kv_heads, n_intermediate) = self._analyze(
                 metadata, prune_heads, prune_ffn, head_ratio, ffn_ratio, uniform_layers)
        except Exception as exc:
            return f"❌ {exc}", "", {}, empty_fig

        head_keep, ffn_keep = self._build_masks(
            head_scores, ffn_scores, n_heads, head_ratio, ffn_ratio, uniform_layers, n_layers)

        total_heads   = n_layers * n_heads
        pruned_heads  = sum(int((~m).sum()) for m in head_keep) if head_keep else 0
        total_neurons = n_layers * n_intermediate if n_intermediate else 0
        pruned_ffn    = sum(int((~m).sum()) for m in ffn_keep) if ffn_keep else 0

        stats = {
            "n_layers":           n_layers,
            "original_heads":     total_heads,
            "pruned_heads":       pruned_heads,
            "head_reduction_pct": round(pruned_heads / total_heads * 100, 1) if total_heads else 0,
            "original_neurons":   total_neurons,
            "pruned_neurons":     pruned_ffn,
            "ffn_reduction_pct":  round(pruned_ffn / total_neurons * 100, 1) if total_neurons else 0,
            "hard_prune":         bool(hard_prune),
        }

        # Layer-0 head importance bar chart
        fig = empty_fig
        if head_scores:
            scores_l0 = head_scores[0]
            keep_l0   = head_keep[0] if head_keep else np.ones(len(scores_l0), dtype=bool)
            colors     = ["#2ecc71" if k else "#e74c3c" for k in keep_l0]
            fig = go.Figure(go.Bar(
                x=[f"H{i}" for i in range(len(scores_l0))],
                y=scores_l0,
                marker_color=colors,
            ))
            fig.update_layout(
                title="Head Importance Scores — Layer 0  (green=kept, red=pruned)",
                xaxis_title="Head", yaxis_title="L2 Norm Score",
                template="plotly_white", height=380,
            )

        md = (
            f"**Preview — no files written**\n\n"
            f"Layers: **{n_layers}**  |  Heads/layer: **{n_heads}**  \n"
            f"Heads to prune: **{pruned_heads}** / {total_heads} "
            f"({stats['head_reduction_pct']:.1f}%)  \n"
            f"FFN neurons to prune: **{pruned_ffn}** / {total_neurons} "
            f"({stats['ffn_reduction_pct']:.1f}%)  \n"
            f"Mode: **{'Hard (structural)' if hard_prune else 'Soft (zero masking)'}**"
        )
        return "✅ Preview complete", md, stats, fig

    def _prune(self, metadata, prune_heads, prune_ffn, hard_prune,
               head_ratio, ffn_ratio, uniform_layers, output_base, storage_fmt):
        import plotly.graph_objects as go

        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        if metadata is None:
            return "❌ No model loaded", "", {}, empty_fig

        model_dir = self._get_model_dir(metadata)
        if model_dir is None:
            return "❌ Cannot resolve model directory", "", {}, empty_fig

        try:
            (config, profile, all_tensors, shard_paths,
             head_scores, ffn_scores,
             n_layers, n_heads, head_dim, n_kv_heads, n_intermediate) = self._analyze(
                 metadata, prune_heads, prune_ffn, head_ratio, ffn_ratio, uniform_layers)
        except Exception as exc:
            return f"❌ {exc}", "", {}, empty_fig

        head_keep, ffn_keep = self._build_masks(
            head_scores, ffn_scores, n_heads, head_ratio, ffn_ratio, uniform_layers, n_layers)

        pruned_tensors: Dict[str, np.ndarray] = dict(all_tensors)

        for layer_idx in range(n_layers):

            # ── Attention heads ──────────────────────────────────────────────
            if head_keep and layer_idx < len(head_keep):
                keep = head_keep[layer_idx]   # bool [n_heads]
                keep_idx = np.where(keep)[0]  # head indices to retain

                def _patch_q(name: str) -> None:
                    w = all_tensors.get(name)
                    if w is None:
                        return
                    if hard_prune:
                        # [n_heads * head_dim, hidden] → keep rows
                        rows = np.concatenate([
                            w[h * head_dim : (h+1) * head_dim]
                            for h in keep_idx
                        ], axis=0)
                        pruned_tensors[name] = rows
                    else:
                        mask = np.ones(w.shape[0], dtype=np.float32)
                        for h in range(n_heads):
                            if not keep[h]:
                                mask[h * head_dim : (h+1) * head_dim] = 0.0
                        pruned_tensors[name] = w * mask[:, None]

                def _patch_o(name: str) -> None:
                    w = all_tensors.get(name)
                    if w is None:
                        return
                    if hard_prune:
                        # [hidden, n_heads * head_dim] → keep columns
                        cols = np.concatenate([
                            w[:, h * head_dim : (h+1) * head_dim]
                            for h in keep_idx
                        ], axis=1)
                        pruned_tensors[name] = cols
                    else:
                        mask = np.ones(w.shape[1], dtype=np.float32)
                        for h in range(n_heads):
                            if not keep[h]:
                                mask[h * head_dim : (h+1) * head_dim] = 0.0
                        pruned_tensors[name] = w * mask[None, :]

                # KV heads for GQA: keep[kv_h] is True if ANY query head
                # that maps to kv_h is kept
                kv_repeat  = n_heads // n_kv_heads if n_kv_heads else 1
                kv_keep    = np.array([
                    any(keep[kv_h * kv_repeat : (kv_h+1) * kv_repeat])
                    for kv_h in range(n_kv_heads)
                ])
                kv_keep_idx = np.where(kv_keep)[0]

                def _patch_kv(name: str) -> None:
                    w = all_tensors.get(name)
                    if w is None:
                        return
                    n_this = w.shape[0] // head_dim  # could be n_kv_heads
                    is_kv  = (n_this == n_kv_heads)
                    local_keep = kv_keep_idx if is_kv else keep_idx
                    if hard_prune:
                        rows = np.concatenate([
                            w[h * head_dim : (h+1) * head_dim]
                            for h in local_keep
                        ], axis=0)
                        pruned_tensors[name] = rows
                    else:
                        mask = np.ones(w.shape[0], dtype=np.float32)
                        for h in range(n_this):
                            local_m = kv_keep if is_kv else keep
                            if not local_m[h]:
                                mask[h * head_dim : (h+1) * head_dim] = 0.0
                        pruned_tensors[name] = w * mask[:, None]

                if profile.fused_qkv_pat:
                    fused_name = profile.fused_qkv_pat.replace("{layer}", str(layer_idx))
                    w = all_tensors.get(fused_name)
                    if w is not None and hard_prune:
                        # Unfuse, slice, re-fuse
                        q_block = np.concatenate([w[h*head_dim:(h+1)*head_dim] for h in keep_idx], 0)
                        k_block = np.concatenate([w[n_heads*head_dim + h*head_dim : n_heads*head_dim + (h+1)*head_dim] for h in keep_idx], 0)
                        v_block = np.concatenate([w[2*n_heads*head_dim + h*head_dim : 2*n_heads*head_dim + (h+1)*head_dim] for h in keep_idx], 0)
                        pruned_tensors[fused_name] = np.concatenate([q_block, k_block, v_block], 0)
                    elif w is not None:
                        # Soft: zero out pruned head slots
                        mask = np.ones(w.shape[0], dtype=np.float32)
                        for h in range(n_heads):
                            if not keep[h]:
                                for offset in (0, n_heads, 2*n_heads):
                                    s = (offset + h) * head_dim
                                    mask[s : s + head_dim] = 0.0
                        pruned_tensors[fused_name] = w * mask[:, None]
                else:
                    _patch_q(profile.q_proj_pat.replace("{layer}", str(layer_idx)))
                    _patch_kv(profile.k_proj_pat.replace("{layer}", str(layer_idx)))
                    _patch_kv(profile.v_proj_pat.replace("{layer}", str(layer_idx)))
                    _patch_o(profile.o_proj_pat.replace("{layer}", str(layer_idx)))

                # Also patch bias tensors if present (replace .weight → .bias)
                for pat in (profile.q_proj_pat, profile.k_proj_pat,
                            profile.v_proj_pat, profile.o_proj_pat):
                    bias_name = pat.replace("{layer}", str(layer_idx)).replace(".weight", ".bias")
                    if bias_name in all_tensors:
                        _patch_q(bias_name)   # bias shape same as weight dim-0

            # ── FFN neurons ──────────────────────────────────────────────────
            if ffn_keep and layer_idx < len(ffn_keep):
                keep_f   = ffn_keep[layer_idx]   # bool [n_intermediate]
                keep_fidx = np.where(keep_f)[0]

                def _patch_up(name: str) -> None:
                    w = all_tensors.get(name)
                    if w is None:
                        return
                    if hard_prune:
                        pruned_tensors[name] = w[keep_fidx]
                    else:
                        mask = keep_f.astype(np.float32)
                        pruned_tensors[name] = w * mask[:, None]

                def _patch_down(name: str) -> None:
                    w = all_tensors.get(name)
                    if w is None:
                        return
                    if hard_prune:
                        pruned_tensors[name] = w[:, keep_fidx]
                    else:
                        mask = keep_f.astype(np.float32)
                        pruned_tensors[name] = w * mask[None, :]

                _patch_up(profile.ffn_up_pat.replace("{layer}", str(layer_idx)))
                if profile.ffn_gate_pat:
                    _patch_up(profile.ffn_gate_pat.replace("{layer}", str(layer_idx)))
                _patch_down(profile.ffn_down_pat.replace("{layer}", str(layer_idx)))

        # Build output path
        model_id = metadata.get("model_id") or model_dir.name
        out_name = make_output_name(model_id, "trans_pr")
        out_dir  = Path(output_base.strip() or "output") / out_name
        out_dir.mkdir(parents=True, exist_ok=True)

        save_model_dir(pruned_tensors, out_dir, dtype_str=str(storage_fmt))
        copy_support_files(model_dir, out_dir)

        # Update config for hard pruning
        cfg_updates: Dict = {}
        if hard_prune:
            if head_keep:
                new_n_heads    = int(head_keep[0].sum()) if head_keep else n_heads
                new_n_kv       = max(1, round(n_kv_heads * new_n_heads / n_heads))
                cfg_updates[profile.n_heads_key]    = new_n_heads
                cfg_updates[profile.n_kv_heads_key] = new_n_kv
                cfg_updates["head_dim"]             = head_dim
            if ffn_keep and ffn_keep[0].size > 0:
                cfg_updates[profile.intermediate_key] = int(ffn_keep[0].sum())
        if cfg_updates:
            update_config(out_dir, cfg_updates)

        # Stats
        total_heads  = n_layers * n_heads
        pruned_heads = sum(int((~m).sum()) for m in head_keep) if head_keep else 0
        total_ffn    = n_layers * n_intermediate if n_intermediate else 0
        pruned_ffn   = sum(int((~m).sum()) for m in ffn_keep) if ffn_keep else 0

        stats = {
            "output_directory":    str(out_dir),
            "hard_prune":          bool(hard_prune),
            "original_heads":      total_heads,
            "pruned_heads":        pruned_heads,
            "head_reduction_pct":  round(pruned_heads / total_heads * 100, 1) if total_heads else 0,
            "original_neurons":    total_ffn,
            "pruned_neurons":      pruned_ffn,
            "ffn_reduction_pct":   round(pruned_ffn / total_ffn * 100, 1) if total_ffn else 0,
        }

        # Head score chart for layer 0
        fig = empty_fig
        if head_scores:
            scores_l0 = head_scores[0]
            keep_l0   = head_keep[0] if head_keep else np.ones(len(scores_l0), dtype=bool)
            colors    = ["#2ecc71" if k else "#e74c3c" for k in keep_l0]
            fig = go.Figure(go.Bar(
                x=[f"H{i}" for i in range(len(scores_l0))],
                y=scores_l0,
                marker_color=colors,
            ))
            fig.update_layout(
                title="Head Importance — Layer 0  (green=kept, red=pruned)",
                xaxis_title="Head", yaxis_title="L2 Norm",
                template="plotly_white", height=380,
            )

        md = (
            f"**Pruning complete ✅**\n\n"
            f"Saved to: `{out_dir}`  \n"
            f"Heads removed: **{pruned_heads}** / {total_heads} "
            f"({stats['head_reduction_pct']:.1f}%)  \n"
            f"FFN neurons removed: **{pruned_ffn}** / {total_ffn} "
            f"({stats['ffn_reduction_pct']:.1f}%)  \n"
            f"Mode: **{'Hard (structural)' if hard_prune else 'Soft (zero masking)'}**\n\n"
            f"Load with:\n"
            f"```python\n"
            f"from transformers import AutoModelForCausalLM\n"
            f"model = AutoModelForCausalLM.from_pretrained('{out_dir}')\n"
            f"```"
        )
        return f"✅ Saved to {out_dir}", md, stats, fig
