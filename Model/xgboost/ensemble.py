import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define target pollutants
targets = ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10', 'NH3']

# Ensure output directory exists
os.makedirs('models/ensemble', exist_ok=True)

# To calculate overall performance
total_r2 = 0
total_norm_rmse = 0

for target in targets:
    target_lower = target.lower()

    # Load data
    X_train = pd.read_csv(f'data/processed/X_train_{target_lower}.csv')
    X_test = pd.read_csv(f'data/processed/X_test_{target_lower}.csv')
    y_train = pd.read_csv(f'data/processed/y_train_{target_lower}.csv').values.ravel()
    y_test = pd.read_csv(f'data/processed/y_test_{target_lower}.csv').values.ravel()

    # Train XGBoost
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Ensemble Prediction
    y_pred_xgb = xgb.predict(X_test)
    y_pred_rf = rf.predict(X_test)
    y_pred = 0.6 * y_pred_xgb + 0.4 * y_pred_rf

    # Evaluation
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Normalized RMSE (using range)
    target_range = y_test.max() - y_test.min()
    norm_rmse = rmse / target_range if target_range != 0 else 0

    total_r2 += r2
    total_norm_rmse += norm_rmse

    logging.info(f" {target} - RMSE: {rmse:.2f}, Normalized RMSE: {norm_rmse:.3f}, R²: {r2:.2f}")

    # Save models
    joblib.dump(xgb, f'xgb_{target_lower}.pkl')
    joblib.dump(rf, f'rf_{target_lower}.pkl')

# Overall metrics
avg_r2 = total_r2 / len(targets)
avg_norm_rmse = total_norm_rmse / len(targets)

logging.info(f"\nOverall Average R²: {avg_r2:.3f}")
logging.info(f"Overall Average Normalized RMSE: {avg_norm_rmse:.3f}")
logging.info("All ensemble models trained, evaluated, and saved successfully.")
