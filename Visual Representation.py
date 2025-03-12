import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from blackscholes import black_scholes

def create_heatmap(param1_range, param2_range, param1_name, param2_name, fixed_params, option_type):
    prices = np.zeros((len(param1_range), len(param2_range)))
    
    for i, p1 in enumerate(param1_range):
        for j, p2 in enumerate(param2_range):
            S = fixed_params["S"] if param1_name != "S" and param2_name != "S" else p1 if param1_name == "S" else p2
            K = fixed_params["K"] if param1_name != "K" and param2_name != "K" else p1 if param1_name == "K" else p2
            T = fixed_params["T"] if param1_name != "T" and param2_name != "T" else p1 if param1_name == "T" else p2
            r = fixed_params["r"] if param1_name != "r" and param2_name != "r" else p1 if param1_name == "r" else p2
            sigma = fixed_params["sigma"] if param1_name != "sigma" and param2_name != "sigma" else p1 if param1_name == "sigma" else p2
            
            price = black_scholes(S, K, T, r, sigma, option_type)
            prices[i, j] = price
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    df = pd.DataFrame(prices, 
                     index=param1_range,
                     columns=param2_range)
    
    sns.heatmap(df, annot=False, cmap="viridis", ax=ax)
    
    ax.set_ylabel(param1_name)
    ax.set_xlabel(param2_name)
    ax.set_title(f"{option_type.capitalize()} Option Price Heatmap")
    
    return fig

def create_model_comparison_plot(S, K, T, r, sigma_range, option_type, ml_model):
    bs_prices = []
    ml_prices = []
    
    for sigma in sigma_range:
        bs_price = black_scholes(S, K, T, r, sigma, option_type)
        ml_price = ml_model.predict(S, K, T, r, sigma)
        
        bs_prices.append(bs_price)
        ml_prices.append(ml_price)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sigma_range, bs_prices, 'b-', label='Black-Scholes')
    ax.plot(sigma_range, ml_prices, 'r--', label='ML Model')
    
    ax.set_xlabel('Volatility (σ)')
    ax.set_ylabel('Option Price')
    ax.set_title(f'{option_type.capitalize()} Option Price: Black-Scholes vs ML Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_feature_importance(importance_dict):
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in ML Model')
    
    return fig