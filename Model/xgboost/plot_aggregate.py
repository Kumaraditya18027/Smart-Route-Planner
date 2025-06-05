import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define targets
targets = ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10', 'NH3']


data = {}
for target in targets:
    data[target] = {
        'X_train': pd.read_csv(f'data/processed/X_train_{target.lower()}.csv'),
        'X_test': pd.read_csv(f'data/processed/X_test_{target.lower()}.csv'),
        'y_train': pd.read_csv(f'data/processed/y_train_{target.lower()}.csv').values.ravel(),
        'y_test': pd.read_csv(f'data/processed/y_test_{target.lower()}.csv').values.ravel()
    }

# Train and collect R²
r2_scores = []
iterations = range(1, 201, 10)
for n in iterations:
    avg_r2 = 0
    for target in targets:
        model = xgb.XGBRegressor(
            n_estimators=n, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        model.fit(data[target]['X_train'], data[target]['y_train'])
        y_pred = model.predict(data[target]['X_test'])
        r2 = r2_score(data[target]['y_test'], y_pred)
        avg_r2 += r2 / len(targets)
    r2_scores.append(avg_r2)
    logging.info(f"Iteration {n}: Average R² = {avg_r2:.2f}")

# Plot line graph
plt.figure(figsize=(10, 6))
plt.plot(iterations, r2_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Average R² (All Parameters)')
plt.title('Aggregate Model Accuracy vs. Training Iterations')
plt.grid(True)
plt.savefig('plots/xgboost_aggregate_accuracy.png')
plt.show()
logging.info("Saved plot to plots/xgboost_aggregate_accuracy.png")