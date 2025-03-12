import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="call"):
    S = max(0.01, S)
    K = max(0.01, K)
    T = max(0.01, T)
    sigma = max(0.01, sigma)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return option_price

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    delta_percent = 0.01
    
    dS = S * delta_percent
    dK = K * delta_percent
    dT = max(0.001, T * delta_percent)
    dr = max(0.0001, r * delta_percent)
    dsigma = sigma * delta_percent
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = theta / 365
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = -((S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = theta / 365
        vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

def generate_training_data(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    
    S = np.random.uniform(50, 200, n_samples)
    K = np.random.uniform(50, 200, n_samples)
    T = np.random.uniform(0.1, 2, n_samples)
    r = np.random.uniform(0.01, 0.05, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)
    
    call_prices = np.array([black_scholes(s, k, t, r_, sig, "call") 
                            for s, k, t, r_, sig in zip(S, K, T, r, sigma)])
    put_prices = np.array([black_scholes(s, k, t, r_, sig, "put") 
                          for s, k, t, r_, sig in zip(S, K, T, r, sigma)])
    
    call_features = np.column_stack((S, K, T, r, sigma))
    put_features = np.column_stack((S, K, T, r, sigma))
    
    return call_features, call_prices, put_features, put_prices