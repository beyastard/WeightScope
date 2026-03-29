"""
WeightScope Plugin - Model Fingerprint & Similarity Score
=========================================================

Exports a compact "fingerprint" of a model's weight distribution as a
fixed-length histogram vector (256 equal-width buckets across the value range).
Two fingerprints can be compared with cosine similarity and L1/L2 distance.

Use cases:
  - Verify that a quantized model is close to its float32 source
  - Detect distribution drift between a base model and a fine-tune
  - Quickly compare two models without loading both simultaneously
  - Build a small library of fingerprints and find the nearest neighbour

Fingerprint JSON format (self-contained, sharable):
  {
    "model_id": "...",
    "timestamp": "...",
    "total_parameters": N,
    "unique_patterns": K,
    "dtypes_found": [...],
    "bucket_edges": [v0, v1, ..., v256],   ← 257 edges for 256 buckets
    "histogram": [c0, c1, ..., c255],      ← weighted counts per bucket
    "histogram_normalized": [p0, ..., p255] ← probabilities (sum = 1.0)
  }

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

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from weightscope.plugins.base import BasePlugin

N_BUCKETS = 256


def _build_fingerprint(df: pd.DataFrame, metadata: Dict, n_buckets: int = N_BUCKETS) -> Dict:
    """Compute a histogram fingerprint from the frequency DataFrame."""
    values = df["value"].values.astype(np.float64)
    counts = df["count"].values.astype(np.float64)
    
    # Normalize range between -1 and 1
    v_min, v_max = float(values.min()), float(values.max())
    if v_min == v_max:
        v_min -= 1.0
        v_max += 1.0
    
    edges  = np.linspace(v_min, v_max, n_buckets + 1)
    hist   = np.zeros(n_buckets, dtype=np.float64)
    
    # Bin each unique value by its weight (count)
    indices = np.searchsorted(edges[1:-1], values, side="right")
    for idx, cnt in zip(indices, counts):
        hist[idx] += cnt

    total = hist.sum()
    hist_norm = (hist / total).tolist() if total > 0 else hist.tolist()
    
    return {
        "model_id":             metadata.get("model_id", "unknown"),
        "timestamp":            datetime.now().isoformat(),
        "total_parameters":     int(metadata.get("total_parameters", 0)),
        "unique_patterns":      int(metadata.get("unique_patterns", 0)),
        "dtypes_found":         metadata.get("dtypes_found", []),
        "shard_count":          metadata.get("shard_count", 1),
        "bucket_edges":         edges.tolist(),
        "histogram":            hist.tolist(),
        "histogram_normalized": hist_norm,
    }


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _l1_distance(a: List[float], b: List[float]) -> float:
    return float(np.sum(np.abs(np.array(a) - np.array(b))))


def _l2_distance(a: List[float], b: List[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _kl_divergence(p: List[float], q: List[float], eps: float = 1e-12) -> float:
    """KL divergence D(P||Q) — measures how P differs from Q."""
    pv = np.array(p) + eps
    qv = np.array(q) + eps
    pv /= pv.sum()
    qv /= qv.sum()
    return float(np.sum(pv * np.log(pv / qv)))


class ModelFingerprintPlugin(BasePlugin):
    name        = "Model Fingerprint"
    version     = "0.1.0"
    description = "Exports a histogram fingerprint and compares two models with cosine similarity."
    
    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("🔏 Fingerprint"):
            gr.Markdown("### Model Fingerprint & Similarity")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Current Model")
                    export_btn    = gr.Button("💾 Export Fingerprint (JSON)", variant="primary")
                    export_status = gr.Textbox(label="Status", interactive=False)
                    fp_plot       = gr.Plot(label="Current Model Fingerprint")

                with gr.Column():
                    gr.Markdown("#### Compare Two Fingerprints")
                    fp_file_a = gr.File(label="Fingerprint A (JSON)")
                    fp_file_b = gr.File(label="Fingerprint B (JSON)")
                    compare_btn   = gr.Button("📐 Compare", variant="secondary")
                    similarity_md = gr.Markdown("*Upload two fingerprint files to compare.*")
                    compare_plot  = gr.Plot(label="Histogram Overlay")
            
            export_btn.click(
                fn=self._export,
                inputs=[self.state["current_df"],
                        self.state["current_metadata"],
                        self.state["current_model_id"]],
                outputs=[export_status, fp_plot],
            )
            
            compare_btn.click(
                fn=self._compare,
                inputs=[fp_file_a, fp_file_b],
                outputs=[similarity_md, compare_plot],
            )
    
    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self, df, metadata, model_id):
        empty_fig = go.Figure().add_annotation(text="No data", showarrow=False)
        if df is None or metadata is None:
            return "❌ No model loaded", empty_fig

        if metadata and model_id:
            metadata = dict(metadata)
            metadata["model_id"] = model_id

        fp = _build_fingerprint(df, metadata or {})

        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)
        safe_id  = str(model_id or "model").replace("/", "--")
        out_path = out_dir / f"{safe_id}_fingerprint.json"
        with open(out_path, "w") as fh:
            json.dump(fp, fh, indent=2)

        fig = self._fingerprint_fig(fp, label=str(model_id or "Model"))
        return f"✅ Saved to {out_path}", fig
    
    # ── Compare ───────────────────────────────────────────────────────────────

    def _compare(self, file_a, file_b):
        empty_fig = go.Figure().add_annotation(text="Upload two fingerprint files", showarrow=False)
        if file_a is None or file_b is None:
            return "*Upload two fingerprint JSON files.*", empty_fig
        
        try:
            with open(file_a.name) as fh:
                fp_a = json.load(fh)
            with open(file_b.name) as fh:
                fp_b = json.load(fh)
        except Exception as exc:
            return f"❌ Could not read files: {exc}", empty_fig
        
        ha  = fp_a.get("histogram_normalized", fp_a.get("histogram", []))
        hb  = fp_b.get("histogram_normalized", fp_b.get("histogram", []))
        
        n   = min(len(ha), len(hb))
        if n == 0:
            return "❌ Empty histograms", empty_fig
        
        ha, hb = ha[:n], hb[:n]

        cos  = _cosine_similarity(ha, hb)
        l1   = _l1_distance(ha, hb)
        l2   = _l2_distance(ha, hb)
        kl   = _kl_divergence(ha, hb)
        kl_r = _kl_divergence(hb, ha)

        name_a = fp_a.get("model_id", "Model A")
        name_b = fp_b.get("model_id", "Model B")
        
        interpretation = (
            "virtually identical"  if cos > 0.999 else
            "very similar"         if cos > 0.99  else
            "similar"              if cos > 0.95  else
            "moderately different" if cos > 0.85  else
            "quite different"
        )
        
        md = f"""
**Model A:** `{name_a}`  ({fp_a.get('total_parameters',0):,} params, {fp_a.get('unique_patterns',0):,} unique)  
**Model B:** `{name_b}`  ({fp_b.get('total_parameters',0):,} params, {fp_b.get('unique_patterns',0):,} unique)  

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Cosine Similarity** | {cos:.6f} | {interpretation} |
| **L1 Distance** | {l1:.6f} | lower = more similar |
| **L2 Distance** | {l2:.6f} | lower = more similar |
| **KL(A‖B)** | {kl:.6f} | information lost approximating A with B |
| **KL(B‖A)** | {kl_r:.6f} | information lost approximating B with A |
"""

        # Overlay histogram
        edges_a = fp_a.get("bucket_edges", list(range(n + 1)))
        edges_b = fp_b.get("bucket_edges", list(range(n + 1)))
        x_a = [(edges_a[i] + edges_a[i+1]) / 2 for i in range(min(n, len(edges_a)-1))]
        x_b = [(edges_b[i] + edges_b[i+1]) / 2 for i in range(min(n, len(edges_b)-1))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_a, y=ha, name=name_a, mode="lines",
                                  line=dict(color="#3498db"), opacity=0.8))
        fig.add_trace(go.Scatter(x=x_b, y=hb, name=name_b, mode="lines",
                                  line=dict(color="#e74c3c", dash="dash"), opacity=0.8))
        fig.update_layout(
            title=f"Histogram Overlay  (cosine similarity = {cos:.4f})",
            xaxis_title="Weight Value",
            yaxis_title="Probability",
            template="plotly_white", height=450,
        )
        return md, fig
    
    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint_fig(fp: Dict, label: str) -> go.Figure:
        edges = fp.get("bucket_edges", [])
        hist  = fp.get("histogram_normalized", fp.get("histogram", []))
        
        n = min(len(hist), len(edges) - 1)
        if n == 0:
            return go.Figure().add_annotation(text="Empty fingerprint", showarrow=False)
        
        x = [(edges[i] + edges[i+1]) / 2 for i in range(n)]
        fig = go.Figure(go.Bar(x=x, y=hist[:n], marker_color="#3498db", opacity=0.8))
        fig.update_layout(
            title=f"Weight Distribution Fingerprint — {label}",
            xaxis_title="Weight Value",
            yaxis_title="Probability",
            template="plotly_white", height=400,
        )
        return fig
