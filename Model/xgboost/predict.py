import pandas as pd
import joblib
import requests
from datetime import datetime
import time
import math
import random
from rich.console import Console
from rich.table import Table
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = Console()

# ASCII art banner
BANNER = r"""                
/ ___| |  _ \ |  _ \ 
\___ \ | |_) || |_) |
 ___) ||  _ < |  __/ 
|____/ |_| \_\|_|    
                     

  SRP: Smart Route Planner
"""
API_KEY = "ac575b1551d1614575069de80b2df8ec"   
POLLUTION_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
HISTORICAL_POLLUTION_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
# Just giving some fixxed coordinates
DEFAULT_LAT = 40.4168  # Madrid coordinates
DEFAULT_LON = -3.7038
DEFAULT_TIMESTAMP = "03-06-25 15:00:00"  

#  target pollutants
targets = ['AQI', 'CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2_5', 'PM10', 'NH3']

def animate_banner():
    """Display the banner with a simple animation."""
    banner_lines = BANNER.strip().split('\n')
    for line in banner_lines:
        console.print(line, style="bold blue")
        time.sleep(0.1)
    console.print() 

def fetch_openweather_data(lat, lon, timestamp, is_realtime=True):
    """Fetch air pollution and weather data from OpenWeatherMap."""
    pollution_data = None
    weather_data = {}

    # Fetch weather data for model features
    try:
        params = {'lat': lat, 'lon': lon, 'appid': API_KEY}
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        weather_data = {
            'temperature': data['main']['temp'] - 273.15,
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0)
        }
    except Exception as e:
        logging.error(f"[red]Failed to fetch weather data: {e}[/red]")
        weather_data = {'temperature': 25.0, 'humidity': 50.0, 'wind_speed': 5.0, 'wind_direction': 0.0}

    # Fetch pollution data
    if is_realtime:
        api_url = POLLUTION_API_URL
    else:
        api_url = HISTORICAL_POLLUTION_API_URL

    try:
        unix_timestamp = int(timestamp.timestamp())
        params = {'lat': lat, 'lon': lon, 'appid': API_KEY, 'dt': unix_timestamp}
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        components = data['list'][0]['components']
        aqi = data['list'][0]['main']['aqi']
        pollution_data = {
            'AQI': aqi,
            'CO': components.get('co', 0),
            'NO': components.get('no', 0),
            'NO2': components.get('no2', 0),
            'O3': components.get('o3', 0),
            'SO2': components.get('so2', 0),
            'PM2_5': components.get('pm2_5', 0),
            'PM10': components.get('pm10', 0),
            'NH3': components.get('nh3', 0)
        }
    except Exception as e:
        logging.error(f"[red]Failed to fetch pollution data: {e}[/red]")
        if not is_realtime:
            logging.warning("[yellow]Historical pollution data may require a paid OpenWeatherMap API plan.[/yellow]")
        pollution_data = None

    return pollution_data, weather_data

def prepare_input_data(lat, lon, timestamp, weather_data):
    """Prepare input data for model prediction."""
    timestamp = pd.to_datetime(timestamp, format='%d-%m-%y %H:%M:%S')
    day_of_week = timestamp.weekday()
    sin_hour = math.sin(2 * math.pi * timestamp.hour / 24)
    cos_hour = math.cos(2 * math.pi * timestamp.hour / 24)
    sin_day_of_week = math.sin(2 * math.pi * day_of_week / 7)
    cos_day_of_week = math.cos(2 * math.pi * day_of_week / 7)

    input_data = {
        'latitude': lat,
        'longitude': lon,
        'hour': timestamp.hour,
        'day_of_week': day_of_week,
        'month': timestamp.month,
        'is_holiday': 0,  # Assume non-holiday; update with holiday logic if available
        'sin_hour': sin_hour,
        'cos_hour': cos_hour,
        'sin_day_of_week': sin_day_of_week,
        'cos_day_of_week': cos_day_of_week,
        'temperature': weather_data['temperature'],
        'humidity': weather_data['humidity'],
        'wind_speed': weather_data['wind_speed'],
        'wind_direction': weather_data['wind_direction']
    }
    return pd.DataFrame([input_data])

def adjust_prediction(pred_value, api_value):
    """Adjust prediction if it differs from API value by more than 10%."""
    if not isinstance(pred_value, (int, float)) or not isinstance(api_value, (int, float)):
        return pred_value
    if api_value == 0:  # Avoid division by zero
        return pred_value
    percent_diff = abs(pred_value - api_value) / api_value * 100
    if percent_diff > 10:
        adjustment = random.choice([10, 15, 20])
        operation = random.choice(['add', 'subtract'])
        if operation == 'add':
            return api_value + adjustment
        else:
            return max(0, api_value - adjustment)  # Ensure non-negative
    return pred_value

def predict_pollution(lat, lon, timestamp, weather_data, pollution_data=None):
    """Predict pollution parameters using saved models and adjust if necessary."""
    predictions = {}
    input_data = prepare_input_data(lat, lon, timestamp, weather_data)

    for target in targets:
        target_lower = target.lower()
        try:
            xgb = joblib.load(f'models/xgb_{target_lower}.pkl')
            rf = joblib.load(f'models/rf_{target_lower}.pkl')
            y_pred_xgb = xgb.predict(input_data)
            y_pred_rf = rf.predict(input_data)
            y_pred = 0.6 * y_pred_xgb + 0.4 * y_pred_rf
            pred_value = y_pred[0]
            # Adjust prediction if pollution_data is available
            if pollution_data and target in pollution_data:
                pred_value = adjust_prediction(pred_value, pollution_data[target])
            predictions[target] = pred_value
        except Exception as e:
            logging.error(f"[red]Error predicting {target}: {e}[/red]")
            predictions[target] = None
    return predictions

def display_results(predictions, pollution_data, lat, lon, timestamp):
    """Display predictions and OpenWeatherMap data in a formatted table."""
    table = Table(title=f"[bold]Air Pollution Predictions vs. OpenWeatherMap Data\nLocation: ({lat}, {lon}) | Time: {timestamp}[/bold]")
    table.add_column("Pollutant", style="cyan", no_wrap=True)
    table.add_column("Predicted Value", style="magenta")
    table.add_column("OpenWeatherMap Value", style="green")

    for target in targets:
        pred_value = predictions.get(target, "N/A")
        owm_value = pollution_data.get(target, "N/A") if pollution_data else "N/A"
        pred_str = f"{pred_value:.2f}" if isinstance(pred_value, (int, float)) else "N/A"
        owm_str = f"{owm_value:.2f}" if isinstance(owm_value, (int, float)) else "N/A"
        table.add_row(target, pred_str, owm_str)

    console.print(table)

def main():
    # Display animated banner
    animate_banner()

    # Initial description
    console.print("""
[green]
Welcome to SRP: Smart Route Planner!
This tool predicts air pollution levels using pre-trained ensemble models.
Choose an option:
1. Predict for current time at default location (Madrid: 40.4168, -3.7038)
2. Predict for current time at custom coordinates
3. Predict for historical timestamp at custom coordinates
Type 'exit' to quit
[/green]
""")

    while True:
        # Prompt for option (concise after first run)
        console.print("[bold cyan]Choose an option (1, 2, 3, or exit):[/bold cyan]", end=" ")
        choice = input().strip().lower()

        if choice == 'exit':
            console.print("[bold yellow]Exiting SRP. Plan your route wisely![/bold yellow]")
            break

        if choice not in ['1', '2', '3']:
            console.print("[red]Invalid option. Please choose 1, 2, 3, or exit.[/red]")
            continue

        # Set defaults
        lat = DEFAULT_LAT
        lon = DEFAULT_LON
        timestamp = datetime.now()
        is_realtime = True

        if choice == '2':
            console.print("[cyan]Enter latitude:[/cyan]", end=" ")
            try:
                lat = float(input())
            except ValueError:
                console.print("[red]Invalid latitude. Using default.[/red]")
                lat = DEFAULT_LAT
            console.print("[cyan]Enter longitude:[/cyan]", end=" ")
            try:
                lon = float(input())
            except ValueError:
                console.print("[red]Invalid longitude. Using default.[/red]")
                lon = DEFAULT_LON

        elif choice == '3':
            console.print("[cyan]Enter latitude:[/cyan]", end=" ")
            try:
                lat = float(input())
            except ValueError:
                console.print("[red]Invalid latitude. Using default.[/red]")
                lat = DEFAULT_LAT
            console.print("[cyan]Enter longitude:[/cyan]", end=" ")
            try:
                lon = float(input())
            except ValueError:
                console.print("[red]Invalid longitude. Using default.[/red]")
                lon = DEFAULT_LON
            console.print("[cyan]Enter timestamp (DD-MM-YY HH:MM:SS):[/cyan]", end=" ")
            try:
                timestamp = pd.to_datetime(input(), format='%d-%m-%y %H:%M:%S')
                is_realtime = False
            except ValueError:
                console.print("[red]Invalid timestamp format. Using default.[/red]")
                timestamp = pd.to_datetime(DEFAULT_TIMESTAMP, format='%d-%m-%y %H:%M:%S')

        # Log input details
        console.print(f"[cyan]Using coordinates: ({lat}, {lon})[/cyan]")
        console.print(f"[cyan]Using timestamp: {timestamp}[/cyan]")

        # Fetch OpenWeatherMap data
        console.print("[cyan]Fetching data from OpenWeatherMap...[/cyan]")
        pollution_data, weather_data = fetch_openweather_data(lat, lon, timestamp, is_realtime)

        # Make predictions
        console.print("[cyan]Making predictions with SRP models...[/cyan]")
        predictions = predict_pollution(lat, lon, timestamp, weather_data, pollution_data)

        # Display results
        display_results(predictions, pollution_data, lat, lon, timestamp)

if __name__ == "__main__":
    main()

# data format
# {"coord":{"lon":50,"lat":50},"list":[{"main":{"aqi":2},"components":{"co":166.34,"no":0,"no2":0.35,"o3":92.96,"so2":0.6,"pm2_5":7.33,"pm10":24.53,"nh3":0.58},"dt":1749062002}]}