import numpy as np
from .base import Payoff


class WorstOfAutocallable(Payoff):
    """
    Prices a Worst-Of Autocallable note with discrete observation dates 
    and a European downside knock-in barrier at maturity.
    """
    
    def __init__(self, 
                 notional: float, 
                 initial_prices: np.ndarray,
                 obs_indices: np.ndarray,       #Time-step indices of observation dates
                 autocall_barriers: np.ndarray, #Barrier levels as % of initial asset price 
                 coupons: np.ndarray,           #Cumulative coupon paid upon autocall
                 dip_barrier: float):           #Downside knock-in barrier 
        
        self.notional=notional
        self.initial_prices=initial_prices
        self.obs_indices=obs_indices
        self.autocall_barriers=autocall_barriers
        self.coupons=coupons
        self.dip_barrier=dip_barrier

    def get_payoffs(self, price_paths: np.ndarray, smoothed: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the path-dependent autocall and maturity logic.
        paths shape: (n_paths, n_assets, n_steps+1)
        Returns discounted payoff samples of shape (n_paths,)
        Has smoothing functionality about the discontinous barrier for Gamma Calculation
        """
        n_paths, n_assets, n_steps_plus_one=price_paths.shape
        maturity_idx=n_steps_plus_one-1
        
        S0=self.initial_prices.reshape(1, n_assets, 1)
        rel_performance=price_paths/S0
        
        #Calculate Worst-Of performance W(t) across the asset axis
        #Shape becomes (n_paths, n_steps+1)
        wo_perf=np.min(rel_performance, axis=1)
     
        #Initialize tracking arrays
        payoff_samples=np.zeros(n_paths)
        payoff_indices=np.full(n_paths,maturity_idx, dtype=int)
        active_paths=np.ones(n_paths, dtype=bool) #True if note hasn't autocalled yet
        
        #Evaluate Autocall Observation Dates
        for i, obs_idx in enumerate(self.obs_indices):
            #Finds paths that hit the barrier AND are still active
            upper_barrier=wo_perf[:, obs_idx]>=self.autocall_barriers[i]
            knock_out=active_paths & upper_barrier
            
            #Payoff
            payoff_samples[knock_out]=self.notional*(1.0+self.coupons[i])
            payoff_indices[knock_out]=obs_idx
            
            #Turns off paths that knocked out
            active_paths=active_paths & ~knock_out
            
        #Maturity logic
        #Only evaluates paths that survived all autocall observation dates
        W_T=wo_perf[:, maturity_idx]
        if not smoothed:
            #Case: wo_perf(T)>=dip_barrier:
            safe_at_maturity=active_paths & (W_T>=self.dip_barrier)
            payoff_samples[safe_at_maturity]=self.notional
        
            #Case: wo_perf(T)<dip_barrier:
            breach_at_maturity=active_paths & ~safe_at_maturity
            payoff_samples[breach_at_maturity]=self.notional*W_T[breach_at_maturity]
        else:
            eps=0.01
            #Computes continuous survival weight [0.0 to 1.0]
            survival_weight=np.clip((W_T-(self.dip_barrier-eps))/(2.0*eps), 0.0, 1.0)
            #Applies the weighted payoff strictly to active paths
            safe_payoff=self.notional*survival_weight
            breach_payoff=self.notional*W_T*(1.0-survival_weight)
            
            payoff_samples[active_paths] = safe_payoff[active_paths] + breach_payoff[active_paths]

        
        return payoff_samples, payoff_indices