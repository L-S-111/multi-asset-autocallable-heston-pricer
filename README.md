# Multi-Asset Autocallable Heston Pricing Engine

This project implements a Multi-Asset Quasi-Monte Carlo pricing and risk engine in Python. It simulates the underlying asset dynamics under the Heston stochastic volatility model, while utilising Andersen's Quadratic-Exponential (QE) discretisation scheme. The engine is specifically designed to evaluate path-dependent, multi-dimensional boundary conditions returning $\mathbb{E}^{\mathbb{Q}}[e^{-r\tau}P(\tau)]$, where $\tau$ is the stopping time of the product--demonstrated here via a Worst-Of Autocallable structure with a European downside knock-in barrier at maturity.

For computational efficiency, we rely entirely on 3D Tensor vectorisation, `(n_paths, n_assets, n_steps)` to avoid iterative loops over simulated paths. Additionally, we employ Quasi-Monte Carlo sequences to achieve low-discrepancy stratification, accelerating convergence alongside exact path-rescaling to compute finite-difference Greeks without requiring SDE re-simulation.

##Theoretical Framework:

### 1. Stochastic Modelling:

The underlying asset price processes are modelled via the multi-dimensional Heston stochastic volatility model. For $n$ assets, the risk-neutral price, $S(t)$, and variance, $V(t)$, for asset $i$ are given by the coupled SDEs:
$$dS_{i}(t)= (r-q_{i})S_{i}(t)dt+\sqrt{V_{i}(t)}S_{i}(t)dW_{i,S}(t)$$
$$dV_{i}(t)=\kappa_{i}(\theta_{i}-V_{i}(t))dt+\xi_{i}\sqrt{V_{i}(t)}dW_{i,V}(t)$$
where $r$ is the risk-free rate, $q_{i}$ is the continuous dividend yield, and $\kappa_{i}, \theta_{i}, \xi_{i}$, are the positive Heston constants. $W_{S}$ and $W_{V}$ are Wiener processes under the risk neutral probability measure $\mathbb{Q}$. The within-asset leverage effect is defined as $\langle dW_{i,S}, dW_{i,V}\rangle=\rho_{i}dt$. The cross-asset price correlation is governed by the correlation matrix $\Sigma$, such that $\langle dW_{i,S}, dW_{j,S}\rangle=\Sigma_{i,j}dt$.

To prevent correlation dampening caused by the independent variance processes, the cross-asset correlation $\Sigma$ is transformed into an implied orthogonal correlation matrix, $\Omega$, where the off-diagonal elements are scaled by:
$$\Omega_{i,j}=\frac{\Sigma_{i,j}}{\sqrt{1-\rho_{i}^{2}}\sqrt{1-\rho_{j}^{2}}}$$
In cases where $\Omega$ loses its positive semi-definiteness, the engine applies a single  spectral projection step to recover the nearest valid correlation matrix before extracting the Cholesky factor $\Omega=LL^{T}$.


### 2. Andersen's Quadratic-Exponential (QE) Discretisation

This engine implements the QE scheme to mitigate discretisation bias and to handle the zero-boundary condition, by matching the first two conditional moments of the variance process, $m_{i}=\mathbb{E}[V_{i}(t+\Delta t)|V_{i}(t)]$ and $s_{i}^{2}=Var[V_{i}(t+\Delta t)|V_{i}(t)]$. Depending on the ratio:
$$\psi_{i}=\frac{s_{i}^{2}}{m_{i}^{2}},$$
the scheme partitions into two regimes determined by the critical threshold $\psi_c \approx 1.5$. The Quadratic regime, with $\psi_{i}\leq\psi_{c}$:
$$V_{i}(t+\Delta t) \approx a_{i}(b_{i} + Z_{i,V})^2, \quad Z_{i,V} \sim \mathcal{N}(0,1)$$
and the Exponential scheme, with $\psi_{i}>\psi_{c}$:
$$V_{i}(t+\Delta t) \approx \Psi^{-1}(U_{i,V}; p_{i}, \beta_{i}), \quad U_{i,V} \sim \mathcal{U}(0,1)$$

### 3. Log Price Discretisation

To avoid the instability from a standard Euler scheme, we employ Andersen's log-price discretisation, based on the exact representation of the integrated variance. The time-integral of the variance process is evaluated via a central discretisation scheme setting $\gamma_{1}=\gamma_{2}=\frac{1}{2}$, yielding the following scheme:
$$\ln S_{i}(t+\Delta t)\approx \ln S_{i}(t)+K_{i,0}+K_{i,1}V_{i}(t)+K_{i,2}V_{i}(t+\Delta t)+\sqrt{K_{i,3}V_{i}(t)+K_{i,4}V_{i}(t+\Delta t)}\cdot Z_{i,S}$$
where $Z_{i,S}\sim\mathcal{N}(0,1)$ is the $i$-th component of the correlated Gaussian vector $\mathbf{Z}_{S}\sim\mathcal{N}(0,\Omega)$. This ensures the price shocks are cross-correlated according to the implied orthogonal matrix, whilst remaining strictly independent of the variance processes. The variables $K_{0}$ through to $K_{4}$ are deterministic constants parameterised by the time-step $\Delta t$ and the underlying Heston parameters. While this scheme does not strictly enforce the discrete-time martingale condition, the net drift away from the martingale measure is negligible under sufficiently small time-steps.

### 4. Quasi-Monte Carlo Integration

To achieve superior asymptotic convergence rates in favour of standard pseudo-random number generation, we map the stochastic drivers to a deterministic, low-discrepancy Sobol sequence. The dimensionality of the integration hypercube is defined as:
$$D=2\times n_{\text{assets}}\times n_{\text{steps}}$$
The generator forces the number of simulated paths to the nearest power of two ($2^{m}$), to preserve the uniform stratification properties of the sequence. Scrambling is applied to permit the computation of valid QMC standard statistical errors.

### 5. Finite Differences via Path Rescaling

The risk sensitive Greeks look at the rate of change of the derivative's value, $P$, with respect to the underlying, with the first order $\Delta$:
$$\Delta=\frac{\partial P}{\partial S_{i}} $$
and second order $\Gamma$:
$$\Gamma=\frac{\partial^{2} P}{\partial S_{i}^{2}}$$
which we approximate via central finite difference. To avoid the expensive re-simulation of the SDE system for every price perturbation, the engine exploits the fact that the Heston price process is homogeneous of degree 1 with respect to $S_{0}$. Hence, the relative performance $\frac{S_{i}(t)}{S_{i}(0)}$ is invariant to shifts in $S_{i}(0)$. By isolating this tensor, shifted paths are generated via a $O(1)$ scalar multiplication, reducing the variance of the finite-difference noise and accelerating risk calculations by orders of magnitude.

## Product Application

### 1. Worst-Of Autocallable
To demonstrate the engine's capability in handling path-dependent, multi-asset boundary conditions, we price a Worst-Of Autocallable note. The worst-of performance of the stocks at any time $t$ is found using:
$$W(t)=\min_{i}(\frac{S_{i}(t)}{S_{i}(0)})$$
The product has a discrete set of observation dates $\Tau=\{t_{1},t_{2},\dots,t_{m}\}$. If $W(t_{k})$ is greater than or equal to the autocall barrier for any $t_{k}\in\tau$, the note auto calls before maturity and pays the notional investment plus a cumulative coupon. If the note survives to maturity $T$ without being called, a European Downside Knock-In barrier is evaluated. If $W(T)$ breaches this barrier, the investor takes a 1-1 loss on their principal mapped to the worst-performing asset.

### 2. Risk Profile and Barrier Smoothing

Boundary conditions, such as the downside knock-in, generate a Heaviside step function at maturity, meaning the theoretical $\Gamma$ is a Dirac delta function. This causes standard finite-difference estimators to exhibit infinite variance at the strike. To compute stable risk sensitivities, the engine implements barrier smoothing by introducing a $2\epsilon$ linear interpolation band $[B-\epsilon, B+\epsilon]$ around the barrier $B$. This maps the binary survival state to a continuous probability weight, capping the maximum $\Delta$ and smoothing the $\Gamma$ profile.

As the worst performing underlying approaches the downside barrier, it causes an abrupt spike and subsequent collapse in $\Gamma$, which can be seen below, where we collect the $\Gamma$ profile of Asset 1 while sweeping across a declining spot price.

![Gamma Profile]](gamma_plot.png)
