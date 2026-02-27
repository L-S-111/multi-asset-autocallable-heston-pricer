import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from typing import Optional


def generate_qmc_drivers(n_paths: int, n_assets: int, n_steps: int, seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates Quasi-Monte Carlo deviates using Sobol sequences for the QE scheme.
    Forces n_paths to the nearest power of 2 to preserve Sobol uniformity properties.
    Returns:
        U_v: Tensor of shape (n_paths, n_assets, n_steps) for variance uniform deviate sampling.
        Z_s: Tensor of shape (n_paths, n_assets, n_steps) for orthogonal price standard normal deviate sampling.
    """
    #dimensionality: deviates for each asset at each step for both variance and price (per path)
    d=2*n_assets*n_steps
    
    #Scrambling applied to the sequence, allowing for RQMC, enabling statistical error estimation while preserving low discrepancy
    sampler=qmc.Sobol(d=d, scramble=True, seed=seed)
    
    #Sobol requires the number of samples to be 2^m
    m=int(np.ceil(np.log2(n_paths)))
    n_samples=2**m
    
    #Generates the sequence (n_samples, d)
    sobol_seq=sampler.random_base2(m=m)
    
    #Reshapes the flat 2D array into a 3D tensor of shape (n_samples, n_steps, 2*n_assets)
    sobol_seq=sobol_seq.reshape(n_samples, n_steps, 2*n_assets)
    
    #Slice into variance and price uniform deviates (n_samples, n_steps, n_assets)
    U_v_raw=sobol_seq[:, :, :n_assets]
    U_s_raw=sobol_seq[:, :, n_assets:]
    
    #Transforms price uniform deviates to standard normal deviates via inverse CDF
    eps=np.finfo(np.float64).eps
    U_s_clipped=np.clip(U_s_raw, eps, 1-eps)
    Z_s_raw = ndtri(U_s_clipped)
    
    #Transpose to (n_samples, n_assets, n_steps) to match the SDE time-loop slicing shape
    U_v=np.transpose(U_v_raw, (0, 2, 1))
    Z_s=np.transpose(Z_s_raw, (0, 2, 1))
    
    return U_v, Z_s