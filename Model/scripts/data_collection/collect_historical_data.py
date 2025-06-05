import requests
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta

# API setup
API_KEY = "35ebeedc4d70fd85a34e9f87c7683ffc"  
POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

# Load coordinates
grid_df = pd.read_csv('../../data/coordinates_grid.csv')

# Define time range
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 1)
time_step = timedelta(hours=1)  # Hourly data

# Initialize SQLite database
conn = sqlite3.connect('../../data/raw/historical_pollution_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS pollution_data (
        latitude REAL, longitude REAL, timestamp INTEGER,
        AQI INTEGER, CO REAL, NO REAL, NO2 REAL, O3 REAL, SO2 REAL,
        PM2_5 REAL, PM10 REAL, NH3 REAL,
        temperature REAL, humidity INTEGER, wind_speed REAL, wind_direction REAL
    )
''')
conn.commit()

# Function to query pollution data
def get_pollution_data(lat, lon, start, end):
    params = {
        'lat': lat, 'lon': lon, 'start': int(start.timestamp()),
        'end': int(end.timestamp()), 'appid': API_KEY
    }
    try:
        response = requests.get(POLLUTION_URL, params=params)
        response.raise_for_status()
        data = response.json()['list']
        return [{
            'timestamp': entry['dt'],
            'AQI': entry['main']['aqi'],
            'CO': entry['components']['co'],
            'NO': entry['components']['no'],
            'NO2': entry['components']['no2'],
            'O3': entry['components']['o3'],
            'SO2': entry['components']['so2'],
            'PM2_5': entry['components']['pm2_5'],
            'PM10': entry['components']['pm10'],
            'NH3': entry['components']['nh3']
        } for entry in data]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pollution data for {lat}, {lon}: {e}")
        return []

# Function to query weather data
def get_weather_data(lat, lon, timestamp):
    params = {'lat': lat, 'lon': lon, 'appid': API_KEY}
    try:
        response = requests.get(WEATHER_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            'timestamp': timestamp,
            'temperature': data['main']['temp'] - 273.15,
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0)
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data for {lat}, {lon}: {e}")
        return None

# Main data collection with interrupt handling
try:
    for _, row in grid_df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        current_time = start_date
        while current_time < end_date:
            chunk_end = min(current_time + timedelta(days=1), end_date)
            pollution_data = get_pollution_data(lat, lon, current_time, chunk_end)
            print(f"Fetching data for lat={lat}, lon={lon}, from {current_time} to {chunk_end}")

            for entry in pollution_data:
                weather_data = get_weather_data(lat, lon, entry['timestamp'])
                if weather_data and weather_data['timestamp'] == entry['timestamp']:
                    combined = {
                        'latitude': lat, 'longitude': lon, 'timestamp': entry['timestamp'],
                        'AQI': entry['AQI'], 'CO': entry['CO'], 'NO': entry['NO'],
                        'NO2': entry['NO2'], 'O3': entry['O3'], 'SO2': entry['SO2'],
                        'PM2_5': entry['PM2_5'], 'PM10': entry['PM10'], 'NH3': entry['NH3'],
                        'temperature': weather_data['temperature'],
                        'humidity': weather_data['humidity'],
                        'wind_speed': weather_data['wind_speed'],
                        'wind_direction': weather_data['wind_direction']
                    }
                    cursor.execute('''
                        INSERT INTO pollution_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', tuple(combined.values()))

            conn.commit()
            current_time = chunk_end
            # time.sleep(1)  # Avoid API rate limits

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user. Saving progress...")

finally:
    conn.commit()  # Save any remaining data
    conn.close()
    print("✅ Database connection closed. Data collection complete.")
