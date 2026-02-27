from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketData:
    """Container for deterministic market parameters."""
    initial_prices: np.ndarray #1D array of initial asset prices S_{i}(0)
    rate: float                #Risk-free rate r  
    dividends: np.ndarray      #1D array of dividend yields q_{i}
    correlation: np.ndarray    #Cross-asset price correlation, nxn matrix

class SDE(ABC):
    """Abstract base class for Stochastic Differential Equations."""
    
    @abstractmethod
    def generate_paths(self, n_paths: int, n_steps: int, T: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Returns a 3D tensor of shape (n_paths, n_assets, n_steps+1) 
        containing the simulated asset paths.
        """
        pass

class Payoff(ABC):
    """Abstract base class for exotic payoffs."""
    
    @abstractmethod
    def get_payoffs(self, price_paths: np.ndarray, smoothed: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the path-dependent logic and returns the undiscounted payoff
        value for each simulated path. Returns a tuple (payoff_samples, payoff_indices).
        """
        pass