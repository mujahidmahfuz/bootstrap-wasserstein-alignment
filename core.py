"""
Bootstrap Wasserstein Alignment (BWA) Core Implementation
"""

import numpy as np
import ot
from scipy import stats
from typing import List, Tuple, Optional
import warnings

class BWA:
    """Bootstrap Wasserstein Alignment for stable feature attribution."""
    
    def __init__(
        self,
        epsilon: float = 0.01,
        max_iter: int = 100,
        tol: float = 1e-6,
        cost_type: str = "correlation",
        random_state: Optional[int] = None
    ):
        """
        Initialize BWA.
        
        Parameters
        ----------
        epsilon : float
            Entropic regularization parameter
        max_iter : int
            Maximum Sinkhorn iterations
        tol : float
            Convergence tolerance
        cost_type : str
            Type of cost matrix: "correlation", "identity", or "euclidean"
        random_state : int, optional
            Random seed for reproducibility
        """
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.cost_type = cost_type
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(
        self,
        replicates: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute BWA consensus from bootstrap replicates.
        
        Parameters
        ----------
        replicates : np.ndarray
            Bootstrap replicates of shape (n_replicates, n_features)
        X : np.ndarray, optional
            Original feature matrix for correlation computation
            
        Returns
        -------
        consensus : np.ndarray
            BWA consensus attribution of shape (n_features,)
        uncertainty : float
            Uncertainty measure (σ_UQ)
        """
        B, d = replicates.shape
        
        # Compute cost matrix
        C = self._compute_cost_matrix(d, X)
        
        # Convert replicates to probability measures
        weights = self._replicates_to_weights(replicates)
        
        # Compute Wasserstein barycenter
        barycenter = ot.barycenter(
            weights, 
            C, 
            reg=self.epsilon,
            numItermax=self.max_iter,
            stopThr=self.tol
        )
        
        # Recover signs
        signs = self._recover_signs(replicates)
        
        # Rescale to preserve energy
        consensus = self._rescale_consensus(
            barycenter, signs, replicates
        )
        
        # Compute uncertainty
        uncertainty = self._compute_uncertainty(replicates)
        
        return consensus, uncertainty
    
    def _compute_cost_matrix(
        self, 
        d: int, 
        X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute cost matrix between features."""
        if self.cost_type == "identity":
            return np.eye(d)
        
        elif self.cost_type == "euclidean":
            indices = np.arange(d).reshape(-1, 1)
            C = ot.dist(indices, indices, metric='euclidean')
            return C / C.max()
        
        elif self.cost_type == "correlation":
            if X is None:
                warnings.warn(
                    "No feature matrix X provided. Using identity cost.",
                    UserWarning
                )
                return np.eye(d)
            
            # Compute correlation-based cost
            corr = np.corrcoef(X, rowvar=False)
            C = 1 - np.abs(corr)
            return C
        
        else:
            raise ValueError(f"Unknown cost type: {self.cost_type}")
    
    def _replicates_to_weights(self, replicates: np.ndarray) -> np.ndarray:
        """Convert replicates to probability measures."""
        weights = np.abs(replicates)
        # Add small constant to avoid division by zero
        weights = weights + 1e-10
        weights = weights / weights.sum(axis=1, keepdims=True)
        return weights
    
    def _recover_signs(self, replicates: np.ndarray) -> np.ndarray:
        """Recover feature signs via binomial testing."""
        B, d = replicates.shape
        signs = np.ones(d)
        
        for j in range(d):
            p_positive = np.mean(replicates[:, j] > 0)
            
            # Binomial test with continuity correction
            z = (p_positive - 0.5) / np.sqrt(0.25 / B)
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            if p_value < 0.05:  # Significant directional preference
                signs[j] = 1 if p_positive > 0.5 else -1
            else:
                # Insufficient evidence - default to positive
                signs[j] = 1
        
        return signs
    
    def _rescale_consensus(
        self, 
        barycenter: np.ndarray, 
        signs: np.ndarray,
        replicates: np.ndarray
    ) -> np.ndarray:
        """Rescale consensus to preserve attribution energy."""
        # Apply signs
        signed_bary = signs * barycenter
        
        # Preserve average norm of replicates
        avg_norm = np.mean([np.linalg.norm(r) for r in replicates])
        current_norm = np.linalg.norm(signed_bary)
        
        if current_norm > 0:
            scaled = signed_bary * (avg_norm / current_norm)
        else:
            scaled = signed_bary
        
        return scaled
    
    def _compute_uncertainty(self, replicates: np.ndarray) -> float:
        """Compute uncertainty estimate σ_UQ."""
        return np.mean(np.std(replicates, axis=0))
    
    def compute_safety_zone(
        self, 
        consensus: np.ndarray, 
        uncertainty: float,
        k: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Safety Zone interval.
        
        Parameters
        ----------
        consensus : np.ndarray
            BWA consensus
        uncertainty : float
            Uncertainty σ_UQ
        k : float
            Multiplier for interval (default: 2 for ~95% coverage)
            
        Returns
        -------
        lower : np.ndarray
            Lower bound of Safety Zone
        upper : np.ndarray
            Upper bound of Safety Zone
        """
        margin = k * uncertainty
        lower = consensus - margin
        upper = consensus + margin
        return lower, upper
