import numpy as np
from scipy.linalg import cholesky, eigh
from .base import SDE, MarketData
from scipy.special import ndtri
from .qmc_drivers import generate_qmc_drivers
from typing import Optional
import warnings
def psi_inv(U_v, p, beta):
        """Vectorized implementation of Psi^{-1}(U_v,p,beta) (Inverse CDF)"""

        #U_v is in [0,1), but clip for numerical safety
        eps=np.finfo(np.float64).eps
        U_v=np.clip(U_v, eps, 1.0-eps)

        #When U_v<=p, the resulting value is <=0.0
        inv_cdf=(np.log1p(-p)-np.log1p(-U_v))/beta
    
        #Clamps negative values to 0.0
        return np.maximum(inv_cdf, 0.0)

def nearest_corr_matrix(Omega: np.ndarray):
     """Runs a single spectral projection step to find a 
        positive semi-definite proxy for an unrealisable correlation matrix"""
     
     eigen_values, eigen_vectors=eigh(Omega)
     eigen_values=np.maximum(eigen_values, 1e-8)

     Omega_pd=eigen_vectors@np.diag(eigen_values)@eigen_vectors.T

     inv_sqrt_diag=1.0/np.sqrt(np.diag(Omega_pd))
     Omega_nearest=Omega_pd*np.outer(inv_sqrt_diag,inv_sqrt_diag)
     return Omega_nearest
     

class MultiAssetHestonQE(SDE):
    """
    Simulates correlated paths for multiple assets under Heston dynamics
    using Andersen's Quadratic-Exponential (QE) discretisation scheme.
    """
    
    def __init__(self, 
                 market_data: MarketData, 
                 v0: np.ndarray,     #Initial variances (n_assets,)
                 kappa: np.ndarray,  #Mean reversion speeds (n_assets,)
                 theta: np.ndarray,  #Long-term variances (n_assets,)
                 xi: np.ndarray,     #Volatility of variance (n_assets,)
                 rho: np.ndarray):   #Within-asset correlation (n_assets,)
        
        self.market=market_data
        self.v0=v0
        self.kappa=kappa
        self.theta=theta
        self.xi=xi 
        self.rho=rho
        
        rho_comp=np.sqrt(1.0-self.rho**2.0)
        denom=np.outer(rho_comp,rho_comp)
        Omega=self.market.correlation/denom
        np.fill_diagonal(Omega,1.0)

        try:
             self.L=cholesky(Omega,lower=True)
        except np.linalg.LinAlgError:
             warnings.warn("Implied orthogonal correlation matrix is unrealisable, using spectral projection step", UserWarning)
             Omega_nearest=nearest_corr_matrix(Omega)
             self.L=cholesky(Omega_nearest, lower=True)

 


    def generate_paths(self, n_paths: int, n_steps: int, T: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Executes the Monte Carlo loop over time steps using tensor vectorisation,
        the QE discretisation scheme for variance and Andersen's log-price discretisation. 
        Accepts a seed to enforce Common Random Numbers.
        """
        n_assets=len(self.market.initial_prices)
        dt=T/n_steps
        
        #Generates deviate drivers upfront, with n_paths rounded up to the nearest 2^m
        U_v_tensor, Z_s_tensor=generate_qmc_drivers(n_paths, n_assets, n_steps, seed=seed)
        n_paths_qmc=U_v_tensor.shape[0]
         
        #Initilise log_price and variance containers accross paths
        log_price_paths=np.zeros((n_paths_qmc, n_assets, n_steps+1))
        variance_paths=np.zeros((n_paths_qmc, n_assets, n_steps+1))
    
        log_price_paths[:, :, 0]=np.log(self.market.initial_prices)
        variance_paths[:, :, 0]=self.v0
       
        #Pre-compute deterministic discretisation constants
        psi_c=1.5
        gamma1=0.5
        gamma2=0.5
        drift=self.market.rate-self.market.dividends

        tiny=np.finfo(np.float64).tiny
        eps=np.finfo(np.float64).eps

        exp_kdt=np.exp(-self.kappa*dt)
        xi2=self.xi**2.0
        rk_xi=(self.rho*self.kappa)/(self.xi)

        K0=drift*dt-rk_xi*self.theta*dt
        K1=gamma1*dt*(rk_xi-0.5)-rk_xi/self.kappa
        K2=gamma2*dt*(rk_xi-0.5)+rk_xi/self.kappa
        K3=gamma1*dt*(1-self.rho**2.0)
        K4=gamma2*dt*(1-self.rho**2.0)


        #Time evolution loop (vectorized across paths and assets)
        for t in range(n_steps):

            v_curr=variance_paths[:,:,t]

            #uniform deviate per (path, asset) for QE variance step t
            U_v=U_v_tensor[:,:,t]

            #Conditional moments of [V_{t+dt}|V_t]  
            m=self.theta+(v_curr-self.theta)*exp_kdt

            s2=(v_curr*xi2*exp_kdt*(1.0-exp_kdt)/self.kappa
                + self.theta*xi2*(1.0-exp_kdt)**2.0/(2.0*self.kappa))

            #protect against divison instability
            m2=np.maximum(m**2.0,tiny)
            psi=s2/m2

            v_next=np.empty_like(v_curr)

            #Case masks
            mask_q=psi<=psi_c
            mask_e=~mask_q
     
            #Quadratic case: psi<=psi_c 
            if np.any(mask_q):

                psi_q=psi[mask_q]
                m_q=m[mask_q]

                root_arg=np.maximum(2/psi_q -1.0, 0.0)
                b2=2.0/psi_q-1.0+np.sqrt(2.0/psi_q)*np.sqrt(root_arg)
                a=m_q/(1+b2)

                Z_v=ndtri(np.clip(U_v[mask_q], eps, 1.0-eps))
                v_next[mask_q]=a*(np.sqrt(b2)+Z_v)**2.0

            #Exponential case: psi>psi_c
            if np.any(mask_e):
                psi_e=psi[mask_e]
                m_e=m[mask_e]

                p=(psi_e-1.0)/(psi_e+1.0)
                beta=2.0/(m_e*(psi_e+1.0))   

                v_next[mask_e]=psi_inv(U_v[mask_e], p, beta)

            #clamps small negative roundoff if any
            v_next=np.maximum(v_next, 0.0)

            variance_paths[:, :, t+1]=v_next    
            
            
            #Price step:

            Z_s_indep=Z_s_tensor[:,:,t]
            #Cross-asset correlated price shocks
            Z_corr=Z_s_indep@self.L.T

            var_term=np.maximum(K3*v_curr+K4*v_next,0.0)

            log_price_paths[:,:, t+1]=(log_price_paths[:, :, t]+K0+K1*v_curr+K2*v_next
                                  +np.sqrt(var_term)*Z_corr)

            
        return np.exp(log_price_paths)