import requests
import pandas as pd
import sqlite3
import time
from datetime import datetime

# API setup
API_KEY = "03786caa01a2b55750050a0074f1b760"
POLLUTION_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
WEATHER_URL = "http://api.openweathermap.org/data/2.5/weather"

# Load coordinates
grid_df = pd.read_csv('../../data/coordinates_grid.csv')

# Initialize SQLite database
conn = sqlite3.connect('../../data/raw/new_realtime_pollution_data.db') 
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

# Function to query current pollution data
def get_current_pollution(lat, lon):
    params = {'lat': lat, 'lon': lon, 'appid': API_KEY}
    try:
        response = requests.get(POLLUTION_URL, params=params)
        response.raise_for_status()
        data = response.json()['list'][0]
        return {
            'timestamp': data['dt'],
            'AQI': data['main']['aqi'],
            'CO': data['components']['co'],
            'NO': data['components']['no'],
            'NO2': data['components']['no2'],
            'O3': data['components']['o3'],
            'SO2': data['components']['so2'],
            'PM2_5': data['components']['pm2_5'],
            'PM10': data['components']['pm10'],
            'NH3': data['components']['nh3']
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pollution data for {lat}, {lon}: {e}")
        return None

# Function to query current weather data
def get_current_weather(lat, lon, timestamp):
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

# Main loop with graceful exit
try:
    while True:
        for _, row in grid_df.iterrows():
            lat, lon = row['latitude'], row['longitude']
            pollution_data = get_current_pollution(lat, lon)
            if pollution_data:
                weather_data = get_current_weather(lat, lon, pollution_data['timestamp'])
                print(f"Fetching data for lat={lat}, lon={lon}")

                if weather_data:
                    combined = {
                        'latitude': lat, 'longitude': lon, 'timestamp': pollution_data['timestamp'],
                        'AQI': pollution_data['AQI'], 'CO': pollution_data['CO'], 'NO': pollution_data['NO'],
                        'NO2': pollution_data['NO2'], 'O3': pollution_data['O3'], 'SO2': pollution_data['SO2'],
                        'PM2_5': pollution_data['PM2_5'], 'PM10': pollution_data['PM10'], 'NH3': pollution_data['NH3'],
                        'temperature': weather_data['temperature'],
                        'humidity': weather_data['humidity'],
                        'wind_speed': weather_data['wind_speed'],
                        'wind_direction': weather_data['wind_direction']
                    }
                    cursor.execute('''
                        INSERT INTO pollution_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', tuple(combined.values()))
        conn.commit()
        print(f"Data collected at {datetime.now()}")
        time.sleep(3600)  # Collect every hour

except KeyboardInterrupt:
    print("\nCtrl+C detected! Saving and exiting...")

finally:
    conn.commit()
    conn.close()
    print("Database connection closed.")
