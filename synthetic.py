#!/usr/bin/env python3
"""
Synthetic Experiment for BWA Paper
Reproduces Table 1 results
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.bwa.core import BWA

def generate_synthetic_data(
    n_samples: int = 20,
    n_features: int = 100,
    sparsity: int = 10,
    block_size: int = 10,
    correlation: float = 0.8,
    random_state: int = 42
) -> tuple:
    """Generate synthetic data with block correlation."""
    np.random.seed(random_state)
    
    # True coefficients (sparse)
    beta_true = np.zeros(n_features)
    active_indices = np.random.choice(
        n_features, sparsity, replace=False
    )
    beta_true[active_indices] = np.random.choice(
        [-2, -1, 1, 2], sparsity, replace=True
    )
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Add block correlation
    for i in range(0, n_features, block_size):
        block_end = min(i + block_size, n_features)
        if block_end - i > 1:
            # Create correlation within block
            base = np.random.randn(n_samples, 1)
            noise = np.random.randn(n_samples, block_end - i - 1) * 0.2
            block = np.hstack([base, base * correlation + noise * np.sqrt(1 - correlation**2)])
            X[:, i:block_end] = block[:, :block_end-i]
    
    # Generate labels
    prob = 1 / (1 + np.exp(-X @ beta_true))
    y = (np.random.rand(n_samples) < prob).astype(int)
    
    return X, y, beta_true

def vanilla_mean(replicates: np.ndarray) -> np.ndarray:
    """Euclidean mean of replicates."""
    return np.mean(replicates, axis=0)

def bootstrap_median(replicates: np.ndarray) -> np.ndarray:
    """Component-wise median."""
    return np.median(replicates, axis=0)

def bootstrapped_shap(replicates: np.ndarray) -> np.ndarray:
    """Bootstrapped SHAP baseline."""
    abs_mean = np.mean(np.abs(replicates), axis=0)
    signs = np.sign(np.median(replicates, axis=0))
    signs[signs == 0] = 1
    return abs_mean * signs

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    active = true != 0
    
    # Sign accuracy (only on active features)
    if np.any(active):
        sign_acc = np.mean(np.sign(pred[active]) == np.sign(true[active]))
    else:
        sign_acc = 0
    
    # Jaccard similarity of top-10 features
    pred_top = np.argsort(np.abs(pred))[-10:]
    true_top = np.argsort(np.abs(true))[-10:]
    jaccard = len(set(pred_top) & set(true_top)) / 10
    
    # MSE
    mse = np.mean((pred - true) ** 2)
    
    # Norm
    norm = np.linalg.norm(pred)
    
    return {
        "sign_accuracy": sign_acc,
        "jaccard": jaccard,
        "mse": mse,
        "norm": norm
    }

def run_experiment(
    n_trials: int = 100,
    n_bootstrap: int = 50,
    n_samples: int = 20,
    n_features: int = 100,
    verbose: bool = True
) -> pd.DataFrame:
    """Main experiment loop."""
    
    methods = {
        "Vanilla Mean": vanilla_mean,
        "Bootstrap Median": bootstrap_median,
        "Bootstrapped SHAP": bootstrapped_shap,
        "BWA": lambda reps: BWA(epsilon=0.01, random_state=42).fit(reps)[0]
    }
    
    all_results = []
    
    if verbose:
        iterator = tqdm(range(n_trials), desc="Running trials")
    else:
        iterator = range(n_trials)
    
    for trial in iterator:
        # Generate data
        X, y, beta_true = generate_synthetic_data(
            n_samples=n_samples,
            n_features=n_features,
            random_state=trial
        )
        
        # Generate bootstrap replicates
        replicates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            # Train logistic regression (simplified)
            # In practice, use sklearn LogisticRegression
            X_boot, y_boot = X[idx], y[idx]
            
            # For demonstration, use analytic solution
            # In real implementation, train actual model
            coef = np.linalg.lstsq(
                X_boot.T @ X_boot + 0.1 * np.eye(n_features),
                X_boot.T @ y_boot,
                rcond=None
            )[0]
            replicates.append(coef)
        
        replicates = np.array(replicates)
        
        # Apply sign-flip instability (Lemma 1)
        mask = np.random.choice(
            [-1, 1], 
            size=replicates.shape, 
            p=[0.2, 0.8]
        )
        replicates *= mask
        
        # Compute metrics for each method
        for method_name, method_func in methods.items():
            pred = method_func(replicates)
            metrics = compute_metrics(pred, beta_true)
            metrics["method"] = method_name
            metrics["trial"] = trial
            all_results.append(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Aggregate results
    summary = df.groupby("method").agg({
        "sign_accuracy": ["mean", "sem"],
        "jaccard": ["mean", "sem"],
        "mse": "mean",
        "norm": "mean"
    }).round(3)
    
    # Format for table
    formatted = []
    for method in methods.keys():
        row = {
            "Method": method,
            "Sign Acc (%)": f"{summary.loc[method, ('sign_accuracy', 'mean')]*100:.1f} "
                          f"± {summary.loc[method, ('sign_accuracy', 'sem')]*100:.1f}",
            "Jaccard@10": f"{summary.loc[method, ('jaccard', 'mean')]:.2f} "
                         f"± {summary.loc[method, ('jaccard', 'sem')]:.02f}",
            "MSE": f"{summary.loc[method, ('mse', 'mean')]:.3f}",
            "∥e∥₂": f"{summary.loc[method, ('norm', 'mean')]:.3f}"
        }
        formatted.append(row)
    
    return pd.DataFrame(formatted)

if __name__ == "__main__":
    print("=" * 70)
    print("Synthetic Stress Test (N=20, d=100)")
    print("=" * 70)
    
    results = run_experiment(
        n_trials=100,
        n_bootstrap=50,
        n_samples=20,
        n_features=100,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("TABLE 1: Synthetic Results")
    print("=" * 70)
    print(results.to_string(index=False))
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/synthetic_results.csv", index=False)
    print("\nResults saved to 'results/synthetic_results.csv'")
