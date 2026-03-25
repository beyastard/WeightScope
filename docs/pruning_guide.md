# Pruning Research Guide

A comprehensive guide to using WeightScope for neural network pruning research.

---

## 🎯 Overview

Pruning reduces model size by removing less important weights. WeightScope provides tools to **identify**, **simulate**, and **analyze** pruning candidates before applying actual modifications.

---

## 📊 Pruning Fundamentals

### What is Pruning?
Original Model: [w₁, w₂, w₃, ..., wₙ] (n parameters)

Pruned Model: [w₁, 0, w₃, ..., 0] (k non-zero parameters, k < n)

### Pruning Granularity

| Type | What's Removed | Sparsity Pattern | Hardware Support |
|------|---------------|------------------|------------------|
| **Unstructured** | Individual weights | Random | Limited (sparse matmul) |
| **Structured** | Neurons/Channels | Regular | Good (smaller matrices) |
| **N:M** | Patterns in blocks | Semi-regular | NVIDIA Ampere+ (2:4) |

**WeightScope Focus**: Unstructured weight analysis (foundation for all pruning types)

---

## 🔍 Pruning Strategies

### 1. Magnitude-Based Pruning (Traditional)

**Principle**: Small weights contribute less to output

**Criterion**: `|w| < ε` (threshold)

**WeightScope Analysis**:
1. Go to **Pruning tab**
2. Adjust "Pruning Threshold (ε)" slider
3. Review "Prunable Parameters" and "Sparsity %"
4. Export candidate list for mask creation

**Typical Thresholds**:
| Model Type | Threshold Range | Expected Sparsity |
|------------|-----------------|-------------------|
| CNN | 1e-4 to 1e-3 | 30-50% |
| Transformer (LLM) | 1e-5 to 1e-4 | 10-30% |
| BERT | 1e-4 to 1e-3 | 40-60% |

**Research Questions**:
- Is magnitude the best importance metric?
- Do small weights recover during fine-tuning?
- What's the optimal threshold per layer?

---

### 2. Frequency-Based Pruning (Novel)

**Principle**: Rare weight values may encode noise or edge cases

**Criterion**: `count(w) ≤ k` (occurrence threshold)

**WeightScope Analysis**:
1. Go to **Query tab** → "🗑️ Low-Count Removal Test"
2. Or **Compression tab** → "Remove Values with Count ≤"
3. Review "Unique Reduction %" and "Param Reduction %"
4. Export for fine-tuning validation

**Example** (AMD-Llama-135m):
Threshold: count ≤ 4
Removed: 26M unique values (37% reduction)
Affected: ~2M parameters (1.5% of total)
Compression gain: Significant for index-based methods

**Research Questions**:
- Do rare values encode important knowledge?
- Is frequency pruning complementary to magnitude pruning?
- Can we combine both for hybrid pruning?

---

### 3. Combined Magnitude + Frequency Pruning

**Principle**: Remove weights that are BOTH small AND rare

**Criterion**: `|w| < ε AND count(w) ≤ k`

**WeightScope Workflow**:
1. Run magnitude pruning analysis (Pruning tab)
2. Run frequency pruning analysis (Compression tab)
3. Export both candidate lists
4. Intersect candidates in Python:
   ```python
   import pandas as pd
   df = pd.read_parquet("model_weights.parquet")
   magnitude_candidates = df[df['value'].abs() < 1e-4]
   frequency_candidates = df[df['count'] <= 4]
   combined = pd.merge(magnitude_candidates, frequency_candidates, how='inner')
   print(f"Combined candidates: {len(combined)}")
   ```
Expected Benefit: More targeted pruning with less accuracy impact

### 4. Clip-Based Pruning
Principle: Extreme outliers may be anomalies worth removing
Criterion: `|w| > T` (clip threshold)

### WeightScope Analysis:
1. Go to Clip & Normalize tab
2. Review "Clipped %" metric
3. Consider clipped weights as pruning candidates
4. Combine with magnitude pruning for outliers + small weights
Use Case: Models with heavy-tailed weight distributions

## 🧪 Pruning Research Workflow

### Phase 1: Baseline Analysis
1. Load model in WeightScope
2. Record baseline metrics:
   - Total parameters
   - Unique value count
   - Weight distribution (min, max, mean, std)
   - Sparsity (|w| < 1e-4)
3. Export baseline distribution

### Phase 2: Candidate Identification
1. Magnitude pruning: Test thresholds 1e-6 to 1e-3
2. Frequency pruning: Test counts 1 to 10
3. Record candidate counts for each strategy
4. Identify overlap between strategies

### Phase 3: Simulation
1. Estimate sparsity for each strategy
2. Calculate expected size reduction
3. Review Query tab results for candidate inspection
4. Select promising strategies for implementation

### Phase 4: Implementation
1. Create pruning mask based on selected criteria
2. Apply mask to model weights
3. Save pruned model (new .safetensors file)
4. Fine-tune for 10-100 epochs

### Phase 5: Evaluation
1. Evaluate on validation set (perplexity, accuracy)
2. Compare pruned vs. baseline
3. Measure actual vs. predicted sparsity
4. Document findings

## 📈 Pruning Metrics Explained

### Sparsity
Sparsity % = (pruned_parameters / total_parameters) × 100

Example:
- Total: 134M parameters
- Pruned: 20M parameters
- Sparsity: 14.9%

### Pruning Ratio
Pruning Ratio = 1 - (remaining_parameters / total_parameters)

Example:
- Remaining: 114M parameters
- Total: 134M parameters
- Ratio: 1 - (114/134) = 14.9%

### Accuracy Impact
Δ Accuracy = (accuracy_pruned - accuracy_baseline) × 100

Interpretation:
- 0% to -1%: Excellent (negligible impact)
- -1% to -3%: Good (acceptable for many applications)
- -3% to -5%: Moderate (fine-tuning recommended)
- < -5%: Poor (reduce pruning ratio)

## 🔬 Advanced Pruning Research

### 1. Iterative Pruning
Approach: Prune → Fine-tune → Prune → Fine-tune (multiple cycles)
WeightScope Role: Analyze weight distribution changes after each cycle
Research Questions:
How many iterations are optimal?
Does distribution converge to a stable state?
Can we predict optimal pruning ratio per iteration?
### 2. Layer-Wise Pruning
Approach: Different pruning ratios for different layers
WeightScope Extension (future feature):
- Per-layer weight distribution analysis
- Layer sensitivity scoring
- Automated layer-wise ratio recommendation
Research Questions:
- Which layers are most sensitive to pruning?
- Can attention layers tolerate more pruning than MLP?
- How does layer depth affect pruning tolerance?
### 3. Pruning + Quantization Co-Optimization
Approach: Jointly optimize pruning and quantization
WeightScope Workflow:
1. Simulate pruning strategies
2. Simulate quantization on pruned distribution
3. Find optimal combination for target size/accuracy
Research Questions:
- Does pruning improve quantization robustness?
- What's the optimal order (prune→quantize vs. quantize→prune)?
- Can we achieve 90% compression with <2% accuracy loss?
### 4. Structured Pruning Analysis
Approach: Extend WeightScope for neuron/channel-level analysis
Future Features:
- Per-neuron weight aggregation
- Channel importance scoring
- Structured pruning mask export
Research Questions:
- How does structured pruning compare to unstructured?
- Can we predict structured pruning impact from weight statistics?
- What's the hardware speedup for different structured patterns?

## 📊 Case Study: AMD-Llama-135m

### Baseline Metrics

**Total Parameters**: 134,105,856

**Unique Values**: 71,260,015

**Weight Range**: [-15.8, +15.8]

**Sparsity (|w| < 1e-4)**: 2.3%

### Magnitude Pruning Results
| Threshold | Prunable Params | Sparsity | Expected Δ Perplexity |
|-----------|-----------------|----------|-----------------------|
| 1e-6 | 500K | 0.4% | < 0.1% |
| 1e-5 | 2M | 1.5% | 0.1-0.3% |
| 1e-4 | 5M | 3.7% | 0.3-0.8% |
| 1e-3 | 15M | 11.2% | 1-3% |

### Frequency Pruning Results
| Count Threshold | Unique Removed | Params Affected | Compression Gain |
|-----------------|----------------|-----------------|------------------|
| ≤ 1 | 15M | 500K | +5% |
| ≤ 4 | 26M | 2M | +12% |
| ≤ 10 | 35M | 5M | +18% |

### Recommended Strategy
1. Magnitude pruning at ε = 1e-4 (3.7% sparsity)
2. Frequency pruning at count ≤ 4 (12% compression gain)
3. Fine-tune for 50 epochs
4. Expected: <1% perplexity increase, 15% size reduction

## 📚 References
1. Han, S. et al. (2015). "Learning both Weights and Connections for Efficient Neural Networks." NeurIPS.
2. Frankle, J. & Carbin, M. (2018). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR.
3. Sun, M. et al. (2019). "Patient Knowledge Distillation for BERT Model Compression." EMNLP.
4. Xia, M. et al. (2022). "Structured Pruning for Large Language Models." arXiv.

## 🤝 Contributing Research
Have you used WeightScope for pruning research? Share your findings:
- Open a GitHub issue with your results
- Submit a pull request to add to this guide
- Cite WeightScope in your publications

---
