#!/usr/bin/env python3
"""
WeightScope: SafeTensors Model Analyzer v0.1.1
A research tool for analyzing model weight distributions, compression potential, and pruning candidates.

Supports: Any .safetensors model (Llama, Mistral, Qwen, Stable Diffusion, Whisper, BERT, etc.)

New Features in v0.1.1:
- Clip & Normalize simulation tab
- Query presets with Custom mode
- Fixed scatter plot filtering (independent masks + stratified sampling)
- Accurate parameter count (tensor-based, not config-based)

Requirements:
    pip install safetensors numpy pandas pyarrow gradio plotly kaleido huggingface_hub psutil numba scipy

Usage:
    python app.py
    # Then open http://127.0.0.1:7860 in your browser

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

# ============================================================================
# IMPORTS
# ============================================================================

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import psutil
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from safetensors import safe_open
from huggingface_hub import model_info, hf_hub_download
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

APP_NAME = "WeightScope: SafeTensors Model Analyzer"
APP_VERSION = "0.1.1"
SAVE_STATE_DIR = Path(".save_state")
DEFAULT_PRUNING_THRESHOLD = 1e-4
MAX_UNIQUE_FOR_PLOT = 100000
MEMORY_SAFETY_THRESHOLD = 0.90

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_model_name(model_id: Union[str, None]) -> str:
    """Convert HF model ID to filesystem-safe name."""
    if model_id is None:
        return "unknown_model"
    if not isinstance(model_id, str):
        model_id = str(model_id)
    return model_id.replace("/", "--")

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for cache validation."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)

def get_total_ram_gb() -> float:
    return psutil.virtual_memory().total / (1024 ** 3)

def format_number(n: int) -> str:
    return f"{n:,}"

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================================
# MODEL LOADING & MEMORY ESTIMATION
# ============================================================================

class ModelLoader:
    """Handles local and remote model loading with memory safety checks."""
    
    def __init__(self):
        self.current_model_path: Optional[Path] = None
        self.current_model_id: Optional[str] = None
        self.is_remote: bool = False
        self.config_data: Optional[Dict] = None
    
    def check_connection(self) -> bool:
        import socket
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            return False
    
    def get_param_count_from_config(self, config_path: Path) -> Optional[int]:
        if not config_path.exists():
            return None
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key in ['num_parameters', 'n_params', 'parameter_count']:
                if key in config:
                    return int(config[key])
            if 'hidden_size' in config and 'num_hidden_layers' in config:
                h = config['hidden_size']
                l = config['num_hidden_layers']
                return int(h * h * l * 12)
        except Exception:
            pass
        return None
    
    def get_param_count_from_hf(self, model_id: str) -> Optional[int]:
        try:
            info = model_info(model_id, timeout=10)
            if hasattr(info, 'safetensors') and info.safetensors:
                return info.safetensors.get('total', None)
            if hasattr(info, 'config') and info.config:
                return info.config.get('num_parameters', None)
        except Exception:
            pass
        return None
    
    def count_params_from_tensors(self, model_path: Path) -> int:
        """Count parameters by reading safetensors header - MOST ACCURATE."""
        try:
            with safe_open(model_path, framework="np") as f:
                total = 0
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    total += tensor.size
                return total
        except Exception:
            return 0
    
    def estimate_memory(self, param_count: int) -> Dict[str, Any]:
        estimated_bytes = param_count * 4 * 1.5
        estimated_gb = estimated_bytes / (1024 ** 3)
        available_gb = get_available_ram_gb()
        
        safe_to_load = estimated_gb < (available_gb * MEMORY_SAFETY_THRESHOLD)
        warning_level = "safe"
        warning_message = ""
        
        if not safe_to_load:
            if estimated_gb > available_gb:
                warning_level = "critical"
                warning_message = f"❌ CRITICAL: Estimated {estimated_gb:.1f} GB exceeds available {available_gb:.1f} GB."
            else:
                warning_level = "warning"
                warning_message = f"⚠️ WARNING: Estimated {estimated_gb:.1f} GB is {estimated_gb/available_gb*100:.0f}% of available RAM."
        elif estimated_gb > 16:
            warning_level = "caution"
            warning_message = f"⚠️ CAUTION: Large model ({estimated_gb:.1f} GB)."
        
        return {
            "param_count": param_count,
            "estimated_gb": round(estimated_gb, 2),
            "available_gb": round(available_gb, 2),
            "safe_to_load": safe_to_load,
            "warning_level": warning_level,
            "warning_message": warning_message
        }
    
    def load_local_model(self, model_dir: str) -> Tuple[bool, str, Optional[Dict]]:
        model_path = Path(model_dir)
        safetensors_file = model_path / "model.safetensors"
        config_file = model_path / "config.json"
        
        if not safetensors_file.exists():
            return False, f"❌ model.safetensors not found in {model_dir}", None
        
        # ✅ PREFER TENSOR COUNT (most accurate)
        param_count = self.count_params_from_tensors(safetensors_file)
        if param_count == 0:
            param_count = self.get_param_count_from_config(config_file)
        
        if param_count == 0:
            return False, "❌ Could not determine parameter count", None
        
        mem_estimate = self.estimate_memory(param_count)
        if not mem_estimate["safe_to_load"]:
            return False, mem_estimate["warning_message"], None
        
        self.current_model_path = safetensors_file
        self.current_model_id = model_path.name
        self.is_remote = False
        
        self.config_data = {}
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            except Exception:
                pass
        
        return True, f"✅ Loaded: {model_path.name} ({format_number(param_count)} params)", mem_estimate
    
    def load_remote_model(self, model_id: str, cache_dir: str = "models") -> Tuple[bool, str, Optional[Dict]]:
        if not self.check_connection():
            return False, "❌ No internet connection. Please use local mode.", None
        
        try:
            info = model_info(model_id, timeout=10)
        except Exception as e:
            return False, f"❌ Model not found: {str(e)}", None
        
        param_count = self.get_param_count_from_hf(model_id)
        
        cache_path = Path(cache_dir) / sanitize_model_name(model_id)
        ensure_dir(cache_path)
        
        try:
            safetensors_path = hf_hub_download(
                repo_id=model_id, filename="model.safetensors",
                cache_dir=str(cache_path.parent), force_download=False
            )
        except Exception as e:
            return False, f"❌ Download failed: {str(e)}", None
        
        config_path = None
        try:
            config_path = hf_hub_download(
                repo_id=model_id, filename="config.json",
                cache_dir=str(cache_path.parent), force_download=False
            )
        except Exception:
            pass
        
        # ✅ PREFER TENSOR COUNT
        if param_count is None:
            if config_path:
                param_count = self.get_param_count_from_config(Path(config_path))
            if param_count is None:
                param_count = self.count_params_from_tensors(Path(safetensors_path))
        
        if param_count == 0:
            return False, "❌ Could not determine parameter count", None
        
        mem_estimate = self.estimate_memory(param_count)
        if not mem_estimate["safe_to_load"]:
            return False, mem_estimate["warning_message"], None
        
        self.current_model_path = Path(safetensors_path)
        self.current_model_id = model_id
        self.is_remote = True
        
        self.config_data = {}
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
            except Exception:
                pass
        
        return True, f"✅ Loaded: {model_id} ({format_number(param_count)} params)", mem_estimate

# ============================================================================
# WEIGHT ANALYSIS ENGINE
# ============================================================================

class WeightAnalyzer:
    """Core analysis engine for model weights."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.model_hash: Optional[str] = None
        self.model_metadata: Optional[Dict] = None
    
    def analyze_model(self, safetensors_path: Path, progress=gr.Progress()) -> Tuple[bool, str]:
        try:
            self.model_hash = compute_file_hash(safetensors_path)
            all_bit_keys = []
            tensor_info = []
            
            with safe_open(safetensors_path, framework="np") as f:
                keys = list(f.keys())
                for idx, key in enumerate(keys):
                    progress(idx / len(keys), desc="Loading tensors")
                    tensor = f.get_tensor(key)
                    
                    if tensor.dtype == np.float32:
                        bit_type = np.uint32
                    elif tensor.dtype == np.float16:
                        bit_type = np.uint16
                    elif tensor.dtype == np.float8_e4m3fn:
                        bit_type = np.uint8
                    else:
                        continue
                    
                    flat_bits = tensor.ravel().view(bit_type)
                    all_bit_keys.append(flat_bits)
                    tensor_info.append({
                        "name": key, "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype), "size": int(tensor.size)
                    })
            
            if not all_bit_keys:
                return False, "❌ No float tensors found"
            
            progress(0.7, desc="Concatenating")
            all_bits = np.concatenate(all_bit_keys)
            
            progress(0.8, desc="Counting unique")
            unique_keys, counts = np.unique(all_bits, return_counts=True)
            float_values = unique_keys.view(np.float32)
            
            progress(0.9, desc="Building table")
            self.df = pd.DataFrame({
                'bit_pattern': [f"0x{k:08X}" for k in unique_keys],
                'value': float_values, 'count': counts, 'bit_key': unique_keys
            })
            
            self.model_metadata = {
                "file_path": str(safetensors_path), "file_hash": self.model_hash,
                "total_parameters": int(counts.sum()), "unique_patterns": len(self.df),
                "tensor_count": len(tensor_info), "tensors": tensor_info,
                "analysis_timestamp": datetime.now().isoformat(), "dtype": "float32"
            }
            
            progress(1.0, desc="Complete")
            return True, f"✅ Analyzed {format_number(self.model_metadata['total_parameters'])} parameters"
            
        except Exception as e:
            return False, f"❌ Analysis failed: {str(e)}"
    
    def get_compression_analysis(self) -> Dict[str, Any]:
        if self.df is None:
            return {}
        unique_count = len(self.df)
        total_params = self.df['count'].sum()
        value_range = self.df['value'].max() - self.df['value'].min()
        
        options = []
        if unique_count <= 65535:
            savings_mb = (32 - 16) / 8 * total_params / (1024 ** 2)
            options.append({
                "method": "uint16_index + lookup_table", "current_bits": 32, "new_bits": 16,
                "compression_ratio": 2.0, "memory_saved_mb": round(savings_mb, 2),
                "loss": "none (lossless)", "feasible": True
            })
        if value_range <= 32:
            precision = value_range / (2 ** 8)
            options.append({
                "method": "fixed_point_Q5_3", "current_bits": 32, "new_bits": 8,
                "compression_ratio": 4.0, "range": f"[{self.df['value'].min():.3f}, {self.df['value'].max():.3f}]",
                "precision_loss": f"±{precision:.4f}", "feasible": True
            })
        sparsity = (self.df['value'].abs() < 1e-3).sum() / len(self.df) * 100
        if sparsity > 30:
            options.append({"method": "sparse_encoding", "description": f"{sparsity:.1f}% near zero", "feasible": True})
        
        return {
            "unique_values": unique_count, "total_parameters": int(total_params),
            "value_range": round(value_range, 4), "sparsity_pct": round(sparsity, 2),
            "compression_options": options
        }
    
    def get_pruning_candidates(self, threshold: float = DEFAULT_PRUNING_THRESHOLD) -> Dict[str, Any]:
        if self.df is None:
            return {}
        candidates = self.df[self.df['value'].abs() < threshold]
        total_params = self.df['count'].sum()
        candidate_params = candidates['count'].sum()
        return {
            "threshold": threshold, "unique_candidates": len(candidates),
            "prunable_parameters": int(candidate_params),
            "sparsity_pct": round(candidate_params / total_params * 100, 3),
            "candidates": candidates
        }
    
    def simulate_quantization(self, bits: int = 8, method: str = "mse") -> Dict[str, Any]:
        """Simulate quantization and calculate error."""
        if self.df is None:
            return {}
        
        values = self.df['value'].values
        counts = self.df['count'].values
        
        v_min, v_max = values.min(), values.max()
        v_range = v_max - v_min
        levels = 2 ** bits
        #step = v_range / levels
        #quantized = np.round((values - v_min) / step) * step + v_min
        
        if v_range > 0:
            step = v_range / levels
            quantized = np.round((values - v_min) / step) * step + v_min
        else:
            # All values are identical - no quantization needed
            step = 0
            quantized = values.copy()  # Return unchanged
        
        error = np.abs(values - quantized)
        
        # Weighted metrics
        mse = np.sum(error ** 2 * counts) / np.sum(counts)
        mae = np.sum(error * counts) / np.sum(counts)
        
        return {
            "bits": bits, "method": method, "mse": float(mse), "mae": float(mae),
            "max_error": float(error.max()), "levels": levels,
            "step_size": float(step), "range": [float(v_min), float(v_max)]
        }
    
    def simulate_low_count_removal(self, max_count: int = 4) -> Dict[str, Any]:
        """Simulate removing weights with count <= max_count."""
        if self.df is None:
            return {}
        total_params = self.df['count'].sum()
        removable = self.df[self.df['count'] <= max_count]
        removed_params = removable['count'].sum()
        removed_unique = len(removable)
        return {
            "removed_parameters": int(removed_params),
            "removed_unique_patterns": int(removed_unique),
            "param_reduction_pct": round(removed_params / total_params * 100, 3),
            "unique_reduction_pct": round(removed_unique / len(self.df) * 100, 2),
            "remaining_params": int(total_params - removed_params),
            "remaining_unique": int(len(self.df) - removed_unique),
            "compression_gain": round((1 - (len(self.df) - removed_unique)/len(self.df)) * 100, 2)
        }
    
    def simulate_clipping_normalization(self, threshold: float, 
                                         normalize_to: Tuple[float, float] = (-1.0, 1.0)) -> Dict[str, Any]:
        """
        Simulate clipping weights to [-threshold, +threshold] then normalizing to target range.
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        values = self.df['value'].values
        counts = self.df['count'].values
        total_params = counts.sum()
        
        # 1. Clip values
        clipped = np.clip(values, -threshold, threshold)
        
        # 2. Normalize to target range
        target_min, target_max = normalize_to
        target_range = target_max - target_min
        clipped_range = 2 * threshold
        
        normalized = (clipped + threshold) / clipped_range * target_range + target_min
        
        # 3. Calculate reconstruction (for loss metrics)
        reconstructed = normalized - target_min
        reconstructed = reconstructed / target_range * clipped_range - threshold
        
        # 4. Compute metrics (weighted by count)
        error = values - reconstructed
        mse = np.sum(error ** 2 * counts) / total_params
        mae = np.sum(np.abs(error) * counts) / total_params
        
        # SNR in dB
        signal_power = np.sum(values ** 2 * counts) / total_params
        noise_power = mse
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Clipping statistics
        clipped_mask = np.abs(values) > threshold
        clipped_params = np.sum(counts[clipped_mask])
        clipped_unique = np.sum(clipped_mask)
        
        # Compression analysis
        original_range = values.max() - values.min()
        new_range = target_range
        
        #bits_saved = max(0, np.log2(original_range / new_range)) if new_range > 0 else 0
        if new_range > 0 and original_range > 0:
            ratio = original_range / new_range
            bits_saved = max(0, np.log2(ratio))
        else:
            bits_saved = 0  # No range = no bits to save
        
        # Unique value analysis after clipping
        unique_after_clip = len(np.unique(clipped))
        
        return {
            "threshold": threshold,
            "target_range": normalize_to,
            "mse": float(mse),
            "mae": float(mae),
            "snr_db": float(snr_db),
            "clipped_parameters": int(clipped_params),
            "clipped_unique_values": int(clipped_unique),
            "clipped_pct": round(clipped_params / total_params * 100, 3),
            "unique_before": len(values),
            "unique_after_clipping": unique_after_clip,
            "unique_reduction_pct": round((1 - unique_after_clip/len(values)) * 100, 2),
            "theoretical_bits_saved": round(bits_saved, 2),
            "compression_gain_pct": round(bits_saved / 32 * 100, 2),
            "original_range": [float(values.min()), float(values.max())],
            "clipped_range": [-threshold, threshold],
            "normalized_range": normalize_to
        }

# ============================================================================
# SESSION CACHING
# ============================================================================

class SessionCache:
    def __init__(self):
        ensure_dir(SAVE_STATE_DIR)
    
    def get_cache_path(self, model_id: Union[str, None]) -> Path:
        sanitized = sanitize_model_name(model_id)
        return SAVE_STATE_DIR / sanitized
    
    def check_cache(self, model_id: Union[str, None], file_hash: str) -> bool:
        if not isinstance(model_id, str):
            return False
        cache_path = self.get_cache_path(model_id)
        state_file = cache_path / "analysis_state.parquet"
        metadata_file = cache_path / "metadata.json"
        if not state_file.exists() or not metadata_file.exists():
            return False
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata.get('file_hash') == file_hash
        except Exception:
            return False
    
    def save_state(self, model_id: str, df: pd.DataFrame, metadata: Dict):
        if not isinstance(model_id, str):
            return
        cache_path = self.get_cache_path(model_id)
        ensure_dir(cache_path)
        ensure_dir(cache_path / "plots")
        export_df = df.drop(columns=['bit_key'], errors='ignore')
        export_df.to_parquet(cache_path / "analysis_state.parquet", index=False)
        with open(cache_path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def load_state(self, model_id: Union[str, None]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        if not isinstance(model_id, str):
            return None, None
        cache_path = self.get_cache_path(model_id)
        state_file = cache_path / "analysis_state.parquet"
        metadata_file = cache_path / "metadata.json"
        if not state_file.exists() or not metadata_file.exists():
            return None, None
        try:
            df = pd.read_parquet(state_file)
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return df, metadata
        except Exception:
            return None, None
    
    def export_data(self, model_id: Union[str, None], output_dir: str, format: str = "parquet") -> str:
        if not isinstance(model_id, str):
            return "❌ Invalid model ID"
        cache_path = self.get_cache_path(model_id)
        state_file = cache_path / "analysis_state.parquet"
        if not state_file.exists():
            return "❌ No analysis data to export"
        output_path = Path(output_dir)
        ensure_dir(output_path)
        try:
            df = pd.read_parquet(state_file)
            export_name = f"{sanitize_model_name(model_id)}_weights"
            if format == "parquet":
                export_file = output_path / f"{export_name}.parquet"
                df.to_parquet(export_file, index=False)
            elif format == "csv":
                export_file = output_path / f"{export_name}.csv"
                df.to_csv(export_file, index=False)
            elif format == "json":
                export_file = output_path / f"{export_name}.json"
                df.to_json(export_file, orient='records', indent=2)
            else:
                return f"❌ Unsupported format: {format}"
            return f"✅ Exported to {export_file}"
        except Exception as e:
            return f"❌ Export failed: {str(e)}"

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_histogram(df: pd.DataFrame, value_min: float, value_max: float, 
                     log_counts: bool = True, nbins: int = 200) -> go.Figure:
    filtered = df[(df['value'] >= value_min) & (df['value'] <= value_max)]
    if len(filtered) == 0:
        return go.Figure().add_annotation(text="No data in selected range", showarrow=False)
    if len(filtered) > MAX_UNIQUE_FOR_PLOT:
        filtered = filtered.sample(n=MAX_UNIQUE_FOR_PLOT, random_state=42)
    fig = px.histogram(filtered, x='value', y='count', nbins=nbins, log_y=log_counts,
                       title=f"Weight Distribution [{value_min:.3f}, {value_max:.3f}]",
                       labels={'value': 'Weight Value', 'count': 'Frequency'})
    fig.update_layout(template='plotly_white', height=500)
    return fig

def create_scatter(df: pd.DataFrame, show_singletons: bool = False, 
                   show_outliers: bool = False) -> go.Figure:
    """✅ FIXED: Independent filter masks + stratified sampling."""
    if df is None or len(df) == 0:
        return go.Figure().add_annotation(text="No data loaded", showarrow=False)
    
    df_plot = df.copy()
    mask = pd.Series([True] * len(df_plot), index=df_plot.index)
    
    # Build masks independently
    if show_singletons and not show_outliers:
        mask = df_plot['count'] == 1
    elif show_outliers and not show_singletons:
        Q1 = df_plot['count'].quantile(0.25)
        Q3 = df_plot['count'].quantile(0.75)
        IQR = Q3 - Q1
        lower = max(0, Q1 - 3 * IQR)
        upper = Q3 + 3 * IQR
        mask = (df_plot['count'] < lower) | (df_plot['count'] > upper)
    elif show_singletons and show_outliers:
        Q1 = df_plot['count'].quantile(0.25)
        Q3 = df_plot['count'].quantile(0.75)
        IQR = Q3 - Q1
        lower = max(0, Q1 - 3 * IQR)
        upper = Q3 + 3 * IQR
        mask = (df_plot['count'] == 1) | (df_plot['count'] < lower) | (df_plot['count'] > upper)
    
    df_plot = df_plot[mask]
    if len(df_plot) == 0:
        return go.Figure().add_annotation(text="No data matching filters", showarrow=False)
    
    # ✅ Stratified sampling: preserve low-count values
    if len(df_plot) > MAX_UNIQUE_FOR_PLOT:
        low_count = df_plot[df_plot['count'] <= 10]
        high_count = df_plot[df_plot['count'] > 10]
        if len(low_count) < MAX_UNIQUE_FOR_PLOT:
            remaining = MAX_UNIQUE_FOR_PLOT - len(low_count)
            high_sampled = high_count.sample(n=min(remaining, len(high_count)), random_state=42)
            df_plot = pd.concat([low_count, high_sampled])
        else:
            df_plot = df_plot.sample(n=MAX_UNIQUE_FOR_PLOT, random_state=42)
    
    fig = px.scatter(df_plot, x='value', y='count', hover_data=['bit_pattern'],
                     log_y=True, title='Value vs. Count (log scale)',
                     color='count', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_white', height=500)
    return fig

def create_comparison_plot(df1: pd.DataFrame, df2: pd.DataFrame, 
                           name1: str = "Model A", name2: str = "Model B") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df1['value'], y=df1['count'], name=name1, opacity=0.6, nbinsx=100))
    fig.add_trace(go.Histogram(x=df2['value'], y=df2['count'], name=name2, opacity=0.6, nbinsx=100))
    fig.update_layout(template='plotly_white', height=500,
                      title='Model Comparison: Weight Distribution',
                      xaxis_title='Weight Value', yaxis_title='Frequency', barmode='overlay')
    return fig

# ============================================================================
# GRADIO UI BUILDERS
# ============================================================================

def create_load_model_ui(loader: ModelLoader, cache: SessionCache):
    def load_model(source: str, local_path: str, hf_model_id: str, progress=gr.Progress()):
        if source == "Local":
            success, message, mem_estimate = loader.load_local_model(local_path)
        else:
            if not hf_model_id.strip():
                return "❌ Please enter a HuggingFace model ID", "", None, None, None, None
            success, message, mem_estimate = loader.load_remote_model(hf_model_id.strip())
        
        if not success:
            return message, "", None, None, None, None
        
        model_id = loader.current_model_id
        if cache.check_cache(model_id, compute_file_hash(loader.current_model_path)):
            message += "\n✅ Loaded from cache"
            df, metadata = cache.load_state(model_id)
            return message, f"Loaded: {model_id}", df, metadata, mem_estimate, model_id
        else:
            analyzer = WeightAnalyzer()
            success, analysis_msg = analyzer.analyze_model(loader.current_model_path, progress)
            if not success:
                return analysis_msg, "", None, None, None, None
            cache.save_state(model_id, analyzer.df, analyzer.model_metadata)
            message += f"\n{analysis_msg}"
            return message, f"Loaded: {model_id}", analyzer.df, analyzer.model_metadata, mem_estimate, model_id
    
    with gr.Tab("📂 Load Model"):
        gr.Markdown("### Select Model Source")
        with gr.Row():
            source_radio = gr.Radio(choices=["Local", "HuggingFace"], value="Local", label="Source")
        with gr.Group():
            local_path = gr.Textbox(label="Local Model Directory", placeholder="C:/models/amd-llama-135m/")
            gr.Markdown("*Enter the full path to your model directory*")
        with gr.Group():
            hf_model_id = gr.Textbox(label="HuggingFace Model ID", placeholder="amd/AMD-Llama-135m")
        load_btn = gr.Button("🚀 Load Model", variant="primary", size="lg")
        load_status = gr.Textbox(label="Status", interactive=False)
        model_info_display = gr.Textbox(label="Model Info", interactive=False)
        mem_estimate_json = gr.JSON(label="Memory Estimate", visible=True)
        
        current_df = gr.State(None)
        current_metadata = gr.State(None)
        current_model_id = gr.State(None)
        
        load_btn.click(fn=load_model, inputs=[source_radio, local_path, hf_model_id],
                      outputs=[load_status, model_info_display, current_df, current_metadata, mem_estimate_json, current_model_id])
    
    return current_df, current_metadata, current_model_id, loader, cache

def create_overview_tab():
    def render_stats(metadata):
        if metadata is None:
            return "❌ No model loaded"
        total = metadata.get('total_parameters', 0)
        unique = metadata.get('unique_patterns', 0)
        tensors = metadata.get('tensor_count', 0)
        return f"""
        ### 📊 Model Overview
        | Metric | Value |
        |--------|-------|
        | **Total Parameters** | {format_number(total)} |
        | **Unique Bit Patterns** | {format_number(unique)} |
        | **Tensor Count** | {tensors} |
        | **Analysis Date** | {metadata.get('analysis_timestamp', 'N/A')[:19]} |
        | **File Hash** | `{metadata.get('file_hash', 'N/A')[:16]}...` |
        """
    with gr.Tab("📊 Overview"):
        stats_md = gr.Markdown("### Load a model to see statistics")
        return stats_md, render_stats

def create_distribution_tab():
    """✅ FIXED: Two separate sliders for min/max range."""
    with gr.Tab("📈 Distribution"):
        gr.Markdown("### Weight Distribution Analysis")
        with gr.Row():
            value_min = gr.Slider(minimum=-20, maximum=20, value=-2, label="Value Min", step=0.1)
            value_max = gr.Slider(minimum=-20, maximum=20, value=2, label="Value Max", step=0.1)
            log_toggle = gr.Checkbox(value=True, label="Log Scale (Counts)")
        hist_plot = gr.Plot(label="Histogram")
        with gr.Row():
            show_single = gr.Checkbox(label="Show Only Singletons (count=1)")
            show_outliers = gr.Checkbox(label="Show Statistical Outliers")
        scatter_plot = gr.Plot(label="Scatter Plot")
        return (value_min, value_max, log_toggle, hist_plot, show_single, show_outliers, scatter_plot)

def create_query_tab():
    """✅ NEW: Query presets dropdown with Custom mode."""
    with gr.Tab("🔎 Query"):
        gr.Markdown("### Filter and Search Weights")
        
        query_preset = gr.Dropdown(
            choices=[
                "Custom",
                "🌱 Pruning Candidates (|v| < 1e-4)",
                "🔍 Singletons (count = 1)",
                "📉 Rare Values (count 1-10)",
                "📈 High-Frequency (count > 10,000)",
                "🎯 Near Zero (|v| < 1e-3)",
                "⚡ Extreme Values (|v| > 5)",
                "🗑️ Low-Count Removal Test (count ≤ 4)"
            ],
            value="Custom", label="Query Preset"
        )
        
        with gr.Group(visible=True) as custom_inputs:
            with gr.Row():
                v_min = gr.Number(value=-2, label="Value Min")
                v_max = gr.Number(value=2, label="Value Max")
                c_min = gr.Number(value=1, label="Count Min")
                c_max = gr.Number(value=100000, label="Count Max")
            search = gr.Textbox(label="Search (bit pattern or value substring)")
        
        query_table = gr.Dataframe(headers=["Bit Pattern", "Value", "Count"],
                                   datatype=["str", "number", "number"], label="Results (top 100)")
        
        def apply_preset(preset):
            if preset == "Custom":
                return (-2, 2, 1, 100000, "", gr.update(visible=True))
            elif "Pruning Candidates" in preset:
                return (-1e-4, 1e-4, 1, 1000000, "", gr.update(visible=False))
            elif "Singletons" in preset:
                return (-20, 20, 1, 1, "", gr.update(visible=False))
            elif "Rare Values" in preset:
                return (-20, 20, 1, 10, "", gr.update(visible=False))
            elif "High-Frequency" in preset:
                return (-20, 20, 10000, 10000000, "", gr.update(visible=False))
            elif "Near Zero" in preset:
                return (-1e-3, 1e-3, 1, 1000000, "", gr.update(visible=False))
            elif "Extreme Values" in preset:
                return (-20, -5, 1, 1000000, "", gr.update(visible=False))
            elif "Low-Count" in preset:
                return (-20, 20, 1, 4, "", gr.update(visible=False))
            return (-2, 2, 1, 100000, "", gr.update(visible=True))
        
        def run_query(df, v_min, v_max, c_min, c_max, search_term, preset):
            if df is None:
                return pd.DataFrame()
            
            if preset != "Custom":
                if "Pruning Candidates" in preset:
                    v_min, v_max, c_min, c_max = -1e-4, 1e-4, 1, 1000000
                elif "Singletons" in preset:
                    v_min, v_max, c_min, c_max = -20, 20, 1, 1
                elif "Rare Values" in preset:
                    v_min, v_max, c_min, c_max = -20, 20, 1, 10
                elif "High-Frequency" in preset:
                    v_min, v_max, c_min, c_max = -20, 20, 10000, 10000000
                elif "Near Zero" in preset:
                    v_min, v_max, c_min, c_max = -1e-3, 1e-3, 1, 1000000
                elif "Extreme Values" in preset:
                    result_neg = df[(df['value'] <= -5) & (df['count'] >= 1) & (df['count'] <= 1000000)]
                    result_pos = df[(df['value'] >= 5) & (df['count'] >= 1) & (df['count'] <= 1000000)]
                    result = pd.concat([result_neg, result_pos])
                    if search_term:
                        result = result[
                            result['bit_pattern'].str.contains(search_term, case=False, na=False) |
                            result['value'].astype(str).str.contains(search_term, na=False)
                        ]
                    return result.sort_values('count', ascending=False).head(100)[['bit_pattern', 'value', 'count']]
                elif "Low-Count" in preset:
                    c_min, c_max = 1, 4
            
            result = df[(df['value'] >= v_min) & (df['value'] <= v_max) &
                       (df['count'] >= c_min) & (df['count'] <= c_max)]
            if search_term:
                result = result[
                    result['bit_pattern'].str.contains(search_term, case=False, na=False) |
                    result['value'].astype(str).str.contains(search_term, na=False)
                ]
            return result.sort_values('count', ascending=False).head(100)[['bit_pattern', 'value', 'count']]
        
        query_btn = gr.Button("🔍 Run Query", variant="secondary")
        
        query_preset.change(fn=apply_preset, inputs=[query_preset],
                           outputs=[v_min, v_max, c_min, c_max, search, custom_inputs])
        
        return (v_min, v_max, c_min, c_max, search, query_table, query_btn, query_preset, custom_inputs, run_query)

def create_compression_tab():
    with gr.Tab("🗜️ Compression"):
        gr.Markdown("### Bit-Width Reduction Analysis")
        compression_md = gr.Markdown("Load a model to see compression options")
        
        quant_bits = gr.Slider(minimum=4, maximum=16, value=8, step=1, label="Quantization Bits")
        quant_method = gr.Dropdown(choices=["mse", "mae"], value="mse", label="Error Metric")
        quant_btn = gr.Button("🧮 Simulate Quantization", variant="secondary")
        quant_results = gr.JSON(label="Quantization Results")
        
        low_count_slider = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Remove Values with Count ≤")
        low_count_btn = gr.Button("🗑️ Simulate Low-Count Removal", variant="secondary")
        low_count_results = gr.JSON(label="Removal Impact")
        
        def simulate_quant(df, bits, method):
            if df is None:
                return {"error": "No data loaded"}
            analyzer = WeightAnalyzer()
            analyzer.df = df
            return analyzer.simulate_quantization(bits, method)
        
        def simulate_removal(df, max_count):
            if df is None:
                return {"error": "No data loaded"}
            analyzer = WeightAnalyzer()
            analyzer.df = df
            return analyzer.simulate_low_count_removal(max_count)
        
        return (compression_md, quant_bits, quant_method, quant_btn, quant_results, 
                low_count_slider, low_count_btn, low_count_results, simulate_quant, simulate_removal)

def create_pruning_tab():
    with gr.Tab("✂️ Pruning"):
        gr.Markdown("### Pruning Candidate Analysis")
        pruning_threshold = gr.Slider(minimum=1e-6, maximum=1e-2, value=DEFAULT_PRUNING_THRESHOLD,
                                      label="Pruning Threshold (ε)", step=1e-6)
        pruning_md = gr.Markdown("Adjust threshold to see pruning candidates")
        pruning_table = gr.Dataframe(headers=["Bit Pattern", "Value", "Count"],
                                     datatype=["str", "number", "number"], label="Pruning Candidates (top 50)")
        
        def analyze_pruning(df, threshold):
            if df is None:
                return "❌ No model loaded", pd.DataFrame()
            analyzer = WeightAnalyzer()
            analyzer.df = df
            results = analyzer.get_pruning_candidates(threshold)
            summary = f"""
            ### Pruning Analysis (ε = {threshold})
            - **Prunable Parameters**: {format_number(results['prunable_parameters'])}
            - **Sparsity**: {results['sparsity_pct']:.3f}%
            - **Unique Candidates**: {format_number(results['unique_candidates'])}
            """
            candidates_df = results['candidates'].head(50)[['bit_pattern', 'value', 'count']]
            return summary, candidates_df
        
        return (pruning_threshold, pruning_md, pruning_table, analyze_pruning)

def create_clip_normalize_tab():
    """✅ NEW: Clip & Normalize simulation tab."""
    with gr.Tab("✂️ Clip & Normalize"):
        gr.Markdown("### Clip Outliers + Normalize to [-1, 1]")
        gr.Markdown("*Simulate the impact of clipping extreme weights and normalizing the distribution*")
        
        with gr.Row():
            threshold_mode = gr.Dropdown(
                choices=["Absolute", "Standard Deviations (σ)", "Percentile"],
                value="Absolute", label="Threshold Mode"
            )
            threshold_value = gr.Slider(
                minimum=0.1, maximum=20, value=5.0, step=0.1,
                label="Threshold Value", info="Absolute: |w| < T; σ: T × std; Percentile: T-th %"
            )
        
        with gr.Row():
            normalize_min = gr.Number(value=-1.0, label="Normalize Min", interactive=False)
            normalize_max = gr.Number(value=1.0, label="Normalize Max", interactive=False)
            gr.Markdown("*Target range is fixed to [-1, 1] for standardization*")
        
        clip_btn = gr.Button("🔬 Simulate Clip + Normalize", variant="primary")
        
        with gr.Accordion("📊 Results", open=True):
            results_json = gr.JSON(label="Metrics")
            with gr.Row():
                mse_plot = gr.Plot(label="Error Distribution")
                range_plot = gr.Plot(label="Value Range Comparison")
        
        gr.Markdown("""
        ### Interpreting Results
        - **MSE/MAE**: Reconstruction error (lower = less information loss)
        - **SNR (dB)**: Signal-to-noise ratio (higher = better preservation)
        - **Clipped %**: Fraction of parameters affected by clipping
        - **Bits Saved**: Theoretical reduction in bits per weight due to smaller range
        - **Unique Reduction**: Fewer unique values → better compression potential
        """)
        
        def run_clip_normalize(df, threshold_mode, threshold_value):
            if df is None or len(df) == 0:
                return {"error": "No data loaded"}, None, None
            
            values = df['value'].values
            counts = df['count'].values
            
            # Determine actual threshold based on mode
            if threshold_mode == "Absolute":
                threshold = threshold_value
            elif threshold_mode == "Standard Deviations (σ)":
                mean = np.sum(values * counts) / np.sum(counts)
                variance = np.sum((values - mean) ** 2 * counts) / np.sum(counts)
                std = np.sqrt(variance)
                threshold = threshold_value * std
            elif threshold_mode == "Percentile":
                sorted_idx = np.argsort(values)
                sorted_vals = values[sorted_idx]
                sorted_counts = counts[sorted_idx]
                cumsum = np.cumsum(sorted_counts)
                percentile_val = sorted_vals[np.searchsorted(cumsum, np.sum(counts) * threshold_value / 100)]
                threshold = abs(percentile_val)
            else:
                threshold = 5.0
            
            analyzer = WeightAnalyzer()
            analyzer.df = df
            results = analyzer.simulate_clipping_normalization(threshold)
            
            # Create plots
            mse_fig = go.Figure()
            mse_fig.add_trace(go.Indicator(
                mode="number+delta",
                value=results['mse'],
                title={"text": "MSE"},
                number={"valueformat": ".2e"}
            ))
            mse_fig.update_layout(height=300)
            
            range_fig = go.Figure()
            range_fig.add_trace(go.Bar(
                x=['Original', 'Clipped', 'Normalized'],
                y=[results['original_range'][1] - results['original_range'][0],
                   results['clipped_range'][1] - results['clipped_range'][0],
                   results['normalized_range'][1] - results['normalized_range'][0]],
                marker_color=['blue', 'orange', 'green']
            ))
            range_fig.update_layout(title='Dynamic Range Reduction', height=300)
            
            return results, mse_fig, range_fig
        
        clip_btn.click(fn=run_clip_normalize, inputs=[gr.State(None), threshold_mode, threshold_value],
                      outputs=[results_json, mse_plot, range_plot])
        
        return threshold_mode, threshold_value, clip_btn, results_json, mse_plot, range_plot, run_clip_normalize

def create_compare_tab():
    with gr.Tab("⚖️ Compare"):
        gr.Markdown("### Compare Two Models")
        gr.Markdown("*Load a second model in a new browser tab, then export both analyses for comparison*")
        compare_file1 = gr.File(label="Model 1 Analysis (Parquet)")
        compare_file2 = gr.File(label="Model 2 Analysis (Parquet)")
        compare_plot = gr.Plot(label="Distribution Comparison")
        
        def run_comparison(file1, file2):
            if file1 is None or file2 is None:
                return go.Figure().add_annotation(text="Upload two analysis files", showarrow=False)
            try:
                df1 = pd.read_parquet(file1.name)
                df2 = pd.read_parquet(file2.name)
                return create_comparison_plot(df1, df2, "Model 1", "Model 2")
            except Exception as e:
                return go.Figure().add_annotation(text=f"Error: {str(e)}", showarrow=False)
        
        compare_btn = gr.Button("📊 Compare", variant="secondary")
        return (compare_file1, compare_file2, compare_plot, compare_btn, run_comparison)

def create_export_tab(cache: SessionCache):
    with gr.Tab("💾 Export"):
        gr.Markdown("### Export Analysis Data")
        export_format = gr.Dropdown(choices=["parquet", "csv", "json"], value="parquet", label="Data Format")
        plot_format = gr.Dropdown(choices=["png", "svg", "html"], value="png", label="Plot Format")
        output_dir = gr.Textbox(label="Output Directory", placeholder="C:/Analysis/Output/", value="./output")
        export_btn = gr.Button("📤 Export Data", variant="primary")
        export_status = gr.Textbox(label="Export Status", interactive=False)
        
        def export_data(model_id, output_dir, data_format):
            if not model_id:
                return "❌ No model loaded"
            return cache.export_data(model_id, output_dir, data_format)
        
        return (export_format, plot_format, output_dir, export_btn, export_status, export_data)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def build_app():
    loader = ModelLoader()
    cache = SessionCache()
    
    with gr.Blocks(title=f"WeightScope v{APP_VERSION}", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # 🔍 WeightScope v{APP_VERSION}
        ### Research tool for weight distribution analysis, compression studies, and pruning evaluation
        
        **Supported Models**: Any `.safetensors` format (Llama, Mistral, Qwen, Stable Diffusion, Whisper, BERT, etc.)
        **Supported Types**: Language, Vision, Audio, Diffusion, Multi-modal, and more
        **Analysis**: Bit-exact value counting, compression potential, pruning candidates
        """)
        
        # Load Model Tab
        current_df, current_metadata, current_model_id, loader, cache = create_load_model_ui(loader, cache)
        
        # Analysis Tabs
        stats_md, update_overview = create_overview_tab()
        value_min, value_max, log_toggle, hist_plot, show_single, show_outliers, scatter_plot = create_distribution_tab()
        (v_min, v_max, c_min, c_max, search, query_table, query_btn, query_preset, custom_inputs, run_query) = create_query_tab()
        (compression_md, quant_bits, quant_method, quant_btn, quant_results, 
         low_count_slider, low_count_btn, low_count_results, simulate_quant, simulate_removal) = create_compression_tab()
        (pruning_threshold, pruning_md, pruning_table, analyze_pruning) = create_pruning_tab()
        (thresh_mode, thresh_val, clip_btn, results_json, mse_plot, range_plot, run_clip_normalize) = create_clip_normalize_tab()
        (compare_file1, compare_file2, compare_plot, compare_btn, run_comparison) = create_compare_tab()
        (export_format, plot_format, output_dir, export_btn, export_status, export_data) = create_export_tab(cache)
        
        # Wire up dependencies
        current_metadata.change(fn=update_overview, inputs=[current_metadata], outputs=[stats_md])
        
        # Query tab
        query_btn.click(fn=lambda df, *args: run_query(df, *args),
                       inputs=[current_df, v_min, v_max, c_min, c_max, search, query_preset],
                       outputs=[query_table])
        
        # Pruning tab
        pruning_threshold.change(fn=analyze_pruning, inputs=[current_df, pruning_threshold],
                                outputs=[pruning_md, pruning_table])
        
        # Compression tab
        quant_btn.click(fn=lambda df, b, m: simulate_quant(df, b, m),
                       inputs=[current_df, quant_bits, quant_method], outputs=[quant_results])
        low_count_btn.click(fn=lambda df, mc: simulate_removal(df, mc),
                           inputs=[current_df, low_count_slider], outputs=[low_count_results])
        
        # Clip & Normalize tab
        clip_btn.click(fn=lambda df, mode, val: run_clip_normalize(df, mode, val),
                      inputs=[current_df, thresh_mode, thresh_val],
                      outputs=[results_json, mse_plot, range_plot])
        
        # Compare tab
        compare_btn.click(fn=run_comparison, inputs=[compare_file1, compare_file2], outputs=[compare_plot])
        
        # Export tab
        export_btn.click(fn=export_data, inputs=[current_model_id, output_dir, export_format], outputs=[export_status])
        
        # Distribution tab event handlers
        def on_histogram_update(df, v_min, v_max, log_toggle):
            if df is None or len(df) == 0:
                return go.Figure().add_annotation(text="No data loaded", showarrow=False)
            vmin = float(v_min)
            vmax = float(v_max)
            if vmin > vmax:
                vmin, vmax = vmax, vmin
            return create_histogram(df, vmin, vmax, log_toggle)
        
        def on_scatter_update(df, show_single, show_outliers):
            if df is None:
                return go.Figure().add_annotation(text="No data loaded", showarrow=False)
            return create_scatter(df, show_single, show_outliers)
        
        value_min.change(fn=on_histogram_update, inputs=[current_df, value_min, value_max, log_toggle], outputs=[hist_plot])
        value_max.change(fn=on_histogram_update, inputs=[current_df, value_min, value_max, log_toggle], outputs=[hist_plot])
        log_toggle.change(fn=on_histogram_update, inputs=[current_df, value_min, value_max, log_toggle], outputs=[hist_plot])
        show_single.change(fn=on_scatter_update, inputs=[current_df, show_single, show_outliers], outputs=[scatter_plot])
        show_outliers.change(fn=on_scatter_update, inputs=[current_df, show_single, show_outliers], outputs=[scatter_plot])
        
        gr.Markdown(f"""
        ---
        **WeightScope v{APP_VERSION}** | SafeTensors Model Analyzer
        - Analysis is cached in `.save_state/` - re-loading the same model is instant
        - For large models (>7B params), ensure you have sufficient RAM (32GB+ recommended)
        - Export data in Parquet format for best performance in downstream analysis
        - Use "Clip & Normalize" to simulate dynamic range reduction for quantization
        """)
    
    return demo

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print(f"DEBUG: Type is {type(APP_VERSION)} and value is {APP_VERSION}")
    print(f"🚀 WeightScope v{APP_VERSION}")
    print(f"📊 SafeTensors Model Analyzer")
    print(f"📁 Cache directory: {SAVE_STATE_DIR.absolute()}")
    print(f"💻 Available RAM: {get_available_ram_gb():.1f} GB / {get_total_ram_gb():.1f} GB total")
    print("-" * 60)
    
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inbrowser=True)
