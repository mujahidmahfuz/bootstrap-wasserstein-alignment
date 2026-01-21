# Bootstrap Wasserstein Alignment (BWA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-ICML%202025-blue.svg)](https://github.com/mujahidmahfuz/bootstrap-wasserstein-alignment/blob/main/paper/paper.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Official implementation of **"Bootstrap Wasserstein Alignment for Stable Feature Attribution in Low-Data Regimes"** (ICML 2025).

## 📖 Overview

**BWA** is a geometric framework that stabilizes feature attributions in low-data regimes ($N \ll d$) by aligning bootstrap replicates via optimal transport. Unlike Euclidean averaging which suffers catastrophic norm collapse (Lemma 3.1), BWA preserves attribution structure while filtering stochastic noise.

<p align="center">
  <img src="figures/mnist_comparison.png" alt="MNIST Comparison" width="800"/>
  <br>
  <em>Figure 1: BWA recovers digit structure while Euclidean mean produces noise</em>
</p>

### 🎯 Key Contributions

1. **Theorem**: Prove Euclidean averaging causes norm collapse in low-data regimes (Lemma 3.1)
2. **Method**: BWA framework using Wasserstein barycenters for geometric consensus
3. **Empirical**: 78% sign accuracy on synthetic data (vs 45% Euclidean) and 35% higher sparsity than SmoothGrad on MNIST
4. **Uncertainty**: Calibrated estimates with 94% empirical coverage

## 📊 Results

### Synthetic Benchmark ($N=20, d=100$)
| Method | Sign Accuracy | Norm Preservation |
|--------|---------------|-------------------|
| Vanilla Mean | 45.2% | 0.082 |
| Bootstrap Median | 58.7% | 0.126 |
| Bootstrapped SHAP | 61.3% | 0.143 |
| **BWA (Ours)** | **78.4%** | **0.487** |

### MNIST Benchmark ($N=100, d=784$)
| Metric | Vanilla IG | SmoothGrad | **BWA** |
|--------|------------|------------|---------|
| Gini Sparsity | 0.412 | 0.556 | **0.684** |
| ∥e∥₂ Preservation | 0.158 | — | **0.892** |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mujahidmahfuz/bootstrap-wasserstein-alignment.git
cd bootstrap-wasserstein-alignment

# Install dependencies
pip install -r requirements.txt

# Install BWA package
pip install -e .
