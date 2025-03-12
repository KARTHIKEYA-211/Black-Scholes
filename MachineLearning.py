import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from blackscholes import generate_training_data

class OptionPricingModel:
    def __init__(self, model_type="call"):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_type = model_type
        self.is_trained = False
        self.model_path = f"option_pricing_{model_type}_model.joblib"
        self.mse = None
        self.r2 = None
    
    def train(self, force_retrain=False):
        if os.path.exists(self.model_path) and not force_retrain:
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            
            call_features, call_prices, put_features, put_prices = generate_training_data(n_samples=1000)
            
            if self.model_type == "call":
                X, y = call_features, call_prices
            else:
                X, y = put_features, put_prices
                
            y_pred = self.model.predict(X)
            self.mse = mean_squared_error(y, y_pred)
            self.r2 = r2_score(y, y_pred)
            
            return True
        
        call_features, call_prices, put_features, put_prices = generate_training_data(n_samples=10000)
        
        if self.model_type == "call":
            X, y = call_features, call_prices
        else:
            X, y = put_features, put_prices
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        y_pred = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        
        joblib.dump(self.model, self.model_path)
        
        return True
    
    def predict(self, S, K, T, r, sigma):
        if not self.is_trained:
            self.train()
        
        features = np.array([[S, K, T, r, sigma]])
        return self.model.predict(features)[0]
    
    def get_performance_metrics(self):
        if not self.is_trained:
            self.train()
        
        if self.mse is None or self.r2 is None:
            call_features, call_prices, put_features, put_prices = generate_training_data(n_samples=1000)
            
            if self.model_type == "call":
                X, y = call_features, call_prices
            else:
                X, y = put_features, put_prices
                
            y_pred = self.model.predict(X)
            self.mse = mean_squared_error(y, y_pred)
            self.r2 = r2_score(y, y_pred)
        
        return {
            "mse": self.mse,
            "r2": self.r2,
            "feature_importance": dict(zip(
                ["S", "K", "T", "r", "sigma"],
                self.model.feature_importances_
            ))
        }