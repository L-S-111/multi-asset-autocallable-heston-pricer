import numpy as np
import matplotlib.pyplot as plt
from heston_engine.base import MarketData
from heston_engine.sde import MultiAssetHestonQE
from heston_engine.payoff import WorstOfAutocallable
from heston_engine.engine import MonteCarloEngine

def main():
    """
    Runs test example on a 3-asset basket, producing the risk-neutral price,
    a standard error estimate and a plot of Gamma for Asset 1
    """
   
    initial_prices=np.array([100.0, 100.0, 100.0])
    rate=0.05
    dividends=np.array([0.02, 0.02, 0.02])

    #Cross-asset price correlation matrix
    correlation=np.array([
        [1.0, 0.7, 0.7],
        [0.7, 1.0, 0.7],
        [0.7, 0.7, 1.0]
    ])

    market_data=MarketData(initial_prices=initial_prices.copy(), rate=rate, dividends=dividends, correlation=correlation)

    #SDE Parameters (satisfying Feller condition 2*kappa*theta>xi^2):
    v0=np.array([0.04, 0.04, 0.04])    
    kappa=np.array([2.0, 2.0, 2.0])    
    theta=np.array([0.04, 0.04, 0.04]) 
    xi=np.array([0.3, 0.3, 0.3])       
    rho=np.array([-0.7, -0.7, -0.7])   

    sde=MultiAssetHestonQE(market_data=market_data, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)

    #Payoff Boundary Conditions (Worst-Of Autocallable)
    T=1.0
    n_steps=252
    obs_indices=np.array([126, 252]) 
    autocall_barriers=np.array([1.0, 1.0])
    coupons=np.array([0.05, 0.10]) 
    dip_barrier=0.60               

    payoff=WorstOfAutocallable(
        notional=100.0, 
        initial_prices=initial_prices.copy(),
        obs_indices=obs_indices, 
        autocall_barriers=autocall_barriers, 
        coupons=coupons, 
        dip_barrier=dip_barrier
    )
    
    engine=MonteCarloEngine(sde=sde, payoff=payoff)

    #Expectation evaluation
    n_paths=100000 
    m=int(np.ceil(np.log2(n_paths)))
    n_samples=2**m
    print(f"n_paths set to {n_samples} for Sobol uniformity")

    P, se, base_paths=engine.price(n_paths=n_paths, n_steps=n_steps, T=T, seed=42)
    print(f"Present Value: {P:.4f} | Standard Error: {se:.4f}\n")

    #Gamma Analysis
    print("Finite difference Delta sweep for Asset 1")
    original_price = engine.sde.market.initial_prices[0]
    n_assets=len(engine.sde.market.initial_prices)
    base_S0 = engine.sde.market.initial_prices.copy().reshape(1, n_assets, 1)
    normalized_paths=base_paths/base_S0
    
    price_range=np.linspace(100.0, 50.0, 21)
    deltas=[]
    Gammas=[]

    #Sweep state variable X_1(t) from 100 down to 50
    for S in price_range:
        engine.sde.market.initial_prices[0]=S
        delta_vec, Gamma_vec=engine.calculate_delta_Gamma(n_steps, T, normalized_paths, perturbation=2e-2)
        deltas.append(delta_vec[0])
        Gammas.append(Gamma_vec[0])
        
    # Reset state
    engine.sde.market.initial_prices[0]=original_price
  
    #Plot 
    fig, ax =plt.subplots(figsize=(10, 6))
    ax.plot(price_range, Gammas, color='red', linewidth=2, marker='o', label='Gamma Profile')
    ax.axvline(x=60.0, color='black', linestyle='--', label='Downside Knock-In Boundary (60%)')
    ax.set_xlabel('Asset 1 Price $S_1(t)$')
    ax.set_ylabel(r'Gamma $\Gamma$')
    ax.set_title('Gamma Profile at Knock-In Boundary')
    
    #Inverts x-axis to read right-to-left as the market value decays
    ax.invert_xaxis()
    ax.grid(True, alpha=0.4)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
 