# bootstrap-wasserstein-alignment
Official implementation of 'Bootstrap Wasserstein Alignment for Stable Feature Attribution in Low-Data Regimes' 


# Bootstrap Wasserstein Alignment (BWA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://github.com/yourusername/bwa-paper/actions/workflows/tests.yml/badge.svg)
![Codecov](https://codecov.io/gh/yourusername/bwa-paper/branch/main/graph/badge.svg)

Official implementation of **"Bootstrap Wasserstein Alignment for Stable Feature Attribution in Low-Data Regimes"** (ICML 2025).

## 🔍 Overview

BWA stabilizes feature attributions in low-data regimes ($N \ll d$) by aligning bootstrap replicates via optimal transport. Unlike Euclidean averaging which suffers catastrophic norm collapse, BWA preserves attribution structure while filtering stochastic noise.

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run synthetic experiment
python src/experiments/synthetic.py

# Run MNIST experiment
python src/experiments/mnist.py



