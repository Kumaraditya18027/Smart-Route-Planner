# scripts/preprocessing/merge_csvs.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

historical_csv = '../../data/processed/processed_historical_pollution_data.csv'
realtime_csv = '../../data/processed/processed_new_realtime_pollution_data.csv'    
try:
    hist_df = pd.read_csv(historical_csv)
    rt_df = pd.read_csv(realtime_csv)
    logging.info(f"Loaded {len(hist_df)} historical rows and {len(rt_df)} real-time rows")
except FileNotFoundError as e:
    logging.error(f"CSV file not found: {e}")
    raise

df = pd.concat([hist_df, rt_df], ignore_index=True)
df = df.drop_duplicates(subset=['latitude', 'longitude', 'timestamp'], keep='last')
logging.info(f"Merged and deduplicated to {len(df)} rows")

df.to_csv('../../data/processed/merged_pollution_data.csv', index=False)
logging.info("Saved to data/processed/merged_pollution_data.csv")