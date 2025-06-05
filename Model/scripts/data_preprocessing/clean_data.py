# scripts/preprocessing/clean_data.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv('../../data/processed/merged_pollution_data.csv')
logging.info(f"Initial rows: {len(df)}")

df = df.dropna(subset=['latitude', 'longitude', 'timestamp'])
logging.info(f"Rows after dropping critical missing: {len(df)}")

numerical_cols = ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10', 'NH3',
                  'temperature', 'humidity', 'wind_speed', 'wind_direction']
df[numerical_cols] = df[numerical_cols].interpolate(method='linear', limit_direction='both')
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
logging.info("Handled missing values")

df.to_csv('../../data/processed/cleaned_pollution_data.csv', index=False)
logging.info("Saved to data/processed/cleaned_pollution_data.csv")