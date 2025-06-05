import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load cleaned data
df = pd.read_csv('../data/processed/cleaned_pollution_data.csv')
logging.info(f"Loaded {len(df)} rows")

# Define features and targets
features = ['latitude', 'longitude', 'hour', 'day_of_week', 'month', 'is_holiday',
            'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week',
            'temperature', 'humidity', 'wind_speed', 'wind_direction']
targets = ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10', 'NH3']

# Prepare X and y
X = df[features]
y = {target: df[target] for target in targets}

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)
logging.info("Scaled numerical features")

# Train-test split for each target
splits = {}
for target in targets:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y[target], test_size=0.2, random_state=42
    )
    splits[target] = (X_train, X_test, y_train, y_test)
    # Save splits
    X_train.to_csv(f'data/processed/X_train_{target.lower()}.csv', index=False)
    X_test.to_csv(f'data/processed/X_test_{target.lower()}.csv', index=False)
    y_train.to_csv(f'data/processed/y_train_{target.lower()}.csv', index=False)
    y_test.to_csv(f'data/processed/y_test_{target.lower()}.csv', index=False)
logging.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Save scaler
joblib.dump(scaler, './scaler.pkl')
logging.info("Saved train/test splits and scaler")