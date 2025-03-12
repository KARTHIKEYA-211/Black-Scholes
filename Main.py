import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from blackscholes import black_scholes, calculate_greeks
from randomforest import OptionPricingModel
from visualizations import create_heatmap, create_model_comparison_plot, plot_feature_importance

st.set_page_config(
    page_title="Black-Scholes Option Pricing Calculator",
    layout="wide"
)

def main():
    st.title("Black-Scholes Option Pricing Calculator with ML")
    
    tabs = st.tabs(["Pricing Calculator", "Model Comparison", "ML Insights"])
    
    with st.sidebar:
        st.header("Input Parameters")
        
        option_type = st.radio("Option Type", ["Call", "Put"])
        
        S = st.slider("Underlying Asset Price (S)", 
                    min_value=10.0, max_value=500.0, value=100.0, step=1.0)
        
        K = st.slider("Strike Price (K)", 
                    min_value=10.0, max_value=500.0, value=100.0, step=1.0)
        
        T = st.slider("Time to Maturity (T) in years", 
                    min_value=0.05, max_value=5.0, value=1.0, step=0.05)
        
        r = st.slider("Risk-Free Interest Rate (r)", 
                    min_value=0.0, max_value=0.2, value=0.03, step=0.005,
                    format="%.3f")
        
        sigma = st.slider("Volatility (σ)", 
                       min_value=0.05, max_value=1.0, value=0.2, step=0.01)
    
    ml_model = OptionPricingModel(model_type=option_type.lower())
    ml_model.train()
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        bs_price = black_scholes(S, K, T, r, sigma, option_type.lower())
        ml_price = ml_model.predict(S, K, T, r, sigma)
        
        with col1:
            st.subheader(f"{option_type} Option Prices")
            
            price_df = pd.DataFrame({
                "Model": ["Black-Scholes", "Machine Learning"],
                "Price": [f"${bs_price:.2f}", f"${ml_price:.2f}"],
                "Difference": ["", f"{((ml_price - bs_price) / bs_price) * 100:.2f}%"]
            })
            
            st.dataframe(price_df)
            
            greeks = calculate_greeks(S, K, T, r, sigma, option_type.lower())
            st.subheader("Greeks")
            greeks_df = pd.DataFrame(greeks, index=["Value"])
            st.dataframe(greeks_df.T)
        
        with col2:
            st.subheader("Black-Scholes Formula")
            st.markdown("""
            **For Call Option:**
            
            $C = S_0 N(d_1) - Ke^{-rT} N(d_2)$
            
            **For Put Option:**
            
            $P = Ke^{-rT} N(-d_2) - S_0 N(-d_1)$
            
            Where:
            - $S_0$ = Current stock price
            - $K$ = Strike price
            - $r$ = Risk-free interest rate
            - $T$ = Time to maturity
            - $\sigma$ = Volatility
            - $N()$ = Cumulative distribution function of standard normal distribution
            - $d_1 = \\frac{\\ln(S_0/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}$
            - $d_2 = d_1 - \\sigma\\sqrt{T}$
            """)
        
        st.header("Option Price Heatmaps")
        
        heatmap_col1, heatmap_col2 = st.columns(2)
        
        with heatmap_col1:
            st.subheader("Strike Price vs. Underlying Price Heatmap")
            
            K_range = np.linspace(max(10, K * 0.5), K * 1.5, 30)
            S_range = np.linspace(max(10, S * 0.5), S * 1.5, 30)
            
            fixed_params = {
                "S": S,
                "K": K,
                "T": T,
                "r": r,
                "sigma": sigma
            }
            
            fig1 = create_heatmap(K_range, S_range, "K", "S", fixed_params, option_type.lower())
            st.pyplot(fig1)
        
        with heatmap_col2:
            st.subheader("Volatility vs. Time to Maturity Heatmap")
            
            sigma_range = np.linspace(max(0.05, sigma * 0.5), min(1.0, sigma * 2.0), 30)
            T_range = np.linspace(max(0.05, T * 0.5), min(5.0, T * 2.0), 30)
            
            fig2 = create_heatmap(sigma_range, T_range, "sigma", "T", fixed_params, option_type.lower())
            st.pyplot(fig2)
    
    with tabs[1]:
        st.header("Black-Scholes vs ML Model Comparison")
        
        st.markdown("""
        This section compares the traditional Black-Scholes model with a Random Forest ML model 
        trained on Black-Scholes generated data. While the ML model should approximate the 
        Black-Scholes, slight differences may emerge due to the learning process and randomness.
        """)
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.subheader("Volatility Comparison")
            sigma_range = np.linspace(0.05, 0.5, 100)
            fig_vol = create_model_comparison_plot(S, K, T, r, sigma_range, option_type.lower(), ml_model)
            st.pyplot(fig_vol)
        
        with comp_col2:
            metrics = ml_model.get_performance_metrics()
            
            st.subheader("Model Performance")
            st.write(f"Mean Squared Error: {metrics['mse']:.6f}")
            st.write(f"R² Score: {metrics['r2']:.6f}")
            
            st.subheader("Feature Importance")
            fig_imp = plot_feature_importance(metrics["feature_importance"])
            st.pyplot(fig_imp)
    
    with tabs[2]:
        st.header("Machine Learning Insights")
        
        st.markdown("""
        The Random Forest model provides insights into option pricing that extend beyond 
        the traditional Black-Scholes model. Here we can explore these insights.
        """)
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.subheader("What If Analysis")
            
            target_price = st.number_input("Target Option Price", 
                                          min_value=0.1, 
                                          max_value=500.0, 
                                          value=bs_price,
                                          step=0.1)
            
            variable_to_solve = st.selectbox("Find value of", 
                                            ["Volatility (σ)", "Strike Price (K)", "Time to Maturity (T)"])
            
            if st.button("Calculate"):
                with st.spinner("Finding solution..."):
                    if variable_to_solve == "Volatility (σ)":
                        value_range = np.linspace(0.01, 1.0, 1000)
                        prices = []
                        
                        for val in value_range:
                            price = ml_model.predict(S, K, T, r, val)
                            prices.append(price)
                        
                        prices = np.array(prices)
                        idx = np.argmin(np.abs(prices - target_price))
                        estimated_value = value_range[idx]
                        
                        st.success(f"Estimated Volatility: {estimated_value:.4f}")
                        
                    elif variable_to_solve == "Strike Price (K)":
                        value_range = np.linspace(max(1, S*0.5), S*1.5, 1000)
                        prices = []
                        
                        for val in value_range:
                            price = ml_model.predict(S, val, T, r, sigma)
                            prices.append(price)
                        
                        prices = np.array(prices)
                        idx = np.argmin(np.abs(prices - target_price))
                        estimated_value = value_range[idx]
                        
                        st.success(f"Estimated Strike Price: {estimated_value:.2f}")
                        
                    elif variable_to_solve == "Time to Maturity (T)":
                        value_range = np.linspace(0.01, 5.0, 1000)
                        prices = []
                        
                        for val in value_range:
                            price = ml_model.predict(S, K, val, r, sigma)
                            prices.append(price)
                        
                        prices = np.array(prices)
                        idx = np.argmin(np.abs(prices - target_price))
                        estimated_value = value_range[idx]
                        
                        st.success(f"Estimated Time to Maturity: {estimated_value:.2f} years")
        
        with insight_col2:
            st.subheader("Option Sensitivities")
            
            sens_fig, sens_ax = plt.subplots(figsize=(10, 6))
            
            param_ranges = {
                "S": np.linspace(S*0.7, S*1.3, 50),
                "K": np.linspace(K*0.7, K*1.3, 50),
                "T": np.linspace(max(0.05, T*0.7), T*1.3, 50),
                "r": np.linspace(max(0.005, r*0.7), min(0.2, r*1.3), 50),
                "sigma": np.linspace(max(0.05, sigma*0.7), min(1.0, sigma*1.3), 50)
            }
            
            params = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma}
            
            for param, range_vals in param_ranges.items():
                prices = []
                
                for val in range_vals:
                    p = params.copy()
                    p[param] = val
                    price = ml_model.predict(p["S"], p["K"], p["T"], p["r"], p["sigma"])
                    prices.append(price)
                
                sens_ax.plot(
                    (range_vals - params[param]) / params[param] * 100, 
                    prices, 
                    label=param
                )
            
            sens_ax.set_xlabel("Parameter Change (%)")
            sens_ax.set_ylabel("Option Price")
            sens_ax.set_title("Option Price Sensitivity to Parameters")
            sens_ax.legend()
            sens_ax.grid(True, alpha=0.3)
            
            st.pyplot(sens_fig)

if __name__ == "__main__":
    main()