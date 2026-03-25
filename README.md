# 🔍 WeightScope

**SafeTensors Model Analyzer** — A research tool for analyzing model weight distributions, compression potential, and pruning candidates.

[![Tests](https://img.shields.io/badge/tests-17%20passed-green)]()
[![Python](https://img.shields.io/badge/python-3.13-blue)]()
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue)]()

## Features

- 📊 **Weight Distribution Analysis** — Histograms, scatter plots, frequency analysis
- 🗜️ **Compression Simulation** — Quantization, bit-width reduction, lookup tables
- ✂️ **Pruning Studies** — Magnitude-based, frequency-based, low-count removal
- ✂️ **Clip & Normalize** — Dynamic range reduction for quantization optimization
- 💾 **Export & Compare** — Parquet/CSV/JSON export, multi-model comparison

## Supported Models

Any `.safetensors` format model:
- **Language**: Llama, Mistral, Qwen, Phi, GPT-Neo, BERT, T5, etc.
- **Vision**: ViT, CLIP, DINO, SAM, etc.
- **Diffusion**: Stable Diffusion 1.x/2.x/XL, etc.
- **Audio**: Whisper, AudioLDM, Bark, etc.
- **Multi-modal**: LLaVA, Fuyu, etc.

## Installation

```bash
git clone https://github.com/beyastard/WeightScope
cd WeightScope
python -m venv .ven
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
The app will open at http://127.0.0.1:7860 in your browser automatically.
