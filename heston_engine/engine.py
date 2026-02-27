import numpy as np
from .base import SDE, Payoff
from typing import Optional, Tuple

class MonteCarloEngine:
    """
    Carries out the Quasi-Monte Carlo simulation, computing the risk-neutral expectation 
    and a proxy for the standard error of the estimator.
    Calculates risk sensitives Deltas and Gammas.
    """
    
    def __init__(self, sde: SDE, payoff: Payoff):
        self.sde=sde
        self.payoff=payoff
        
    def price(self, n_paths: int, n_steps: int, T: float, seed: Optional[int] = None) -> Tuple[float, float, np.ndarray]:
        """
        Executes the simulation and returns the present value estimation and Standard Error and the raw price_paths.
        Passes the seed to the SDE for path replication.
        """
        dt=T/n_steps
        #Price paths, shape: (n_paths, n_assets, n_steps+1)
        price_paths=self.sde.generate_paths(n_paths, n_steps, T, seed=seed)
        
        #Discounted Payoff samples, shape: (n_paths,)
        payoff_samples, payoff_indices=self.payoff.get_payoffs(price_paths, smoothed=False)

        df=np.exp(-self.sde.market.rate*payoff_indices*dt)
        discounted_payoffs=payoff_samples*df
        
        #Expectation of discounted payoff
        P=float(np.mean(discounted_payoffs))
        
        #Standard Error of the Monte Carlo estimator (conservative estimate for QMC)
        se=float(np.std(discounted_payoffs, ddof=1)/np.sqrt(len(payoff_samples)))
        
        return P, se, price_paths

    def calculate_delta_Gamma(self, n_steps: int, T: float, normalised_paths: np.ndarray, perturbation: float=2e-2) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the first-order risk sensitivity (Delta) for each asset in the basket 
        using central finite differences and path-rescaling (homogeneity of degree 1).
        Returns a tuple of shape ((n_assets,),(n_assets)),
        and uses Common Random Numbers (CRN).
        """
        n_assets=len(self.sde.market.initial_prices)
        dt=T/n_steps
        r=self.sde.market.rate
    
        base_paths=normalised_paths*self.sde.market.initial_prices.reshape(1, n_assets, 1)
        payoffs_base, payoff_idx_base=self.payoff.get_payoffs(base_paths, smoothed=True)
        P_base=float(np.mean(payoffs_base*np.exp(-r*payoff_idx_base*dt)))

        deltas=np.zeros(n_assets)
        Gammas=np.zeros(n_assets)
           
        for i in range(n_assets):
            P_shifted=np.zeros(2)
            dS=np.maximum(self.sde.market.initial_prices[i]*perturbation, 1e-4)
            for j, shift_val in enumerate([dS,-dS]):
                S0_shifted=self.sde.market.initial_prices.copy()
                S0_shifted[i]+=shift_val

                shifted_paths=normalised_paths*S0_shifted.reshape(1,n_assets,1)
                
                payoff_samples, payoff_indices=self.payoff.get_payoffs(shifted_paths, smoothed=True)
                P_shifted[j]=float(np.mean(payoff_samples*np.exp(-r*payoff_indices*dt)))
    
            
            deltas[i]=(P_shifted[0]-P_shifted[1])/(2.0*dS)
            Gammas[i]=(P_shifted[0]-2.0*P_base+P_shifted[1])/(dS**2.0)
            
        return deltas, Gammas