import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import holidays

# Connect to database
conn = sqlite3.connect('../../data/raw/new_realtime_pollution_data.db')

# Load data
df = pd.read_sql_query("SELECT * FROM pollution_data", conn)
conn.close()

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

# Add temporal features
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['datetime'].dt.month
df['is_holiday'] = df['datetime'].apply(
    lambda x: 1 if x in holidays.Spain(years=x.year) else 0  # Adjust for your country
)

# Cyclical encoding for hour and day_of_week
df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Save to CSV
df.to_csv('../../data/processed/processed_new_realtime_pollution_data.csv', index=False)
print("Data processed and saved to processed_pollution_data.csv")