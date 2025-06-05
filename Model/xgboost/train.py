import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define targets
targets = ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10', 'NH3']
units = {'AQI': '', 'CO': 'µg/m³', 'NO': 'µg/m³', 'NO2': 'µg/m³', 'O3': 'µg/m³',
         'SO2': 'µg/m³', 'PM2_5': 'µg/m³', 'PM10': 'µg/m³', 'NH3': 'µg/m³'}

# Train and evaluate models
models = {}
for target in targets:
    # Load data
    X_train = pd.read_csv(f'data/processed/X_train_{target.lower()}.csv')
    X_test = pd.read_csv(f'data/processed/X_test_{target.lower()}.csv')
    y_train = pd.read_csv(f'data/processed/y_train_{target.lower()}.csv').values.ravel()
    y_test = pd.read_csv(f'data/processed/y_test_{target.lower()}.csv').values.ravel()

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logging.info(f"{target} - RMSE: {rmse:.2f} {units[target]}, R²: {r2:.2f}")
    
    # Save model
    models[target] = model
    joblib.dump(model, f'./{target.lower()}.pkl')

logging.info("Saved all models to models/xgboost/")