import requests
import argparse
from datetime import datetime
from typing import Dict, Any, Union, Optional
import json

class WeatherAPI:
    """
    A class to interact with the OpenWeatherMap API to fetch weather data
    for a given city or zip code.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the WeatherAPI with your API key.
        
        Args:
            api_key: Your OpenWeatherMap API key
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, location: str, is_zip: bool = False, country_code: str = "us") -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location: City name or zip code
            is_zip: True if location is a zip code, False if it's a city name
            country_code: Two-letter country code (default: "us")
            
        Returns:
            Dictionary containing weather data
        """
        endpoint = f"{self.base_url}/weather"
        
        params = {
            "appid": self.api_key,
            "units": "imperial"  # For Fahrenheit, use "metric" for Celsius
        }
        
        if is_zip:
            params["zip"] = f"{location},{country_code}"
        else:
            params["q"] = f"{location},{country_code}"
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get("message", "Unknown error")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")
    
    def get_forecast(self, location: str, is_zip: bool = False, country_code: str = "us", days: int = 5) -> Dict[str, Any]:
        """
        Get weather forecast for a location.
        
        Args:
            location: City name or zip code
            is_zip: True if location is a zip code, False if it's a city name
            country_code: Two-letter country code (default: "us")
            days: Number of days for forecast (default: 5)
            
        Returns:
            Dictionary containing forecast data
        """
        endpoint = f"{self.base_url}/forecast"
        
        params = {
            "appid": self.api_key,
            "units": "imperial",  # For Fahrenheit, use "metric" for Celsius
            "cnt": min(days * 8, 40)  # API returns data in 3-hour increments, max 5 days (40 time points)
        }
        
        if is_zip:
            params["zip"] = f"{location},{country_code}"
        else:
            params["q"] = f"{location},{country_code}"
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = response.json().get("message", "Unknown error")
            raise Exception(f"API Error ({response.status_code}): {error_msg}")


def format_current_weather(weather_data: Dict[str, Any]) -> str:
    """
    Format current weather data into a readable string.
    
    Args:
        weather_data: Weather data dictionary from the API
        
    Returns:
        Formatted weather information string
    """
    city = weather_data["name"]
    country = weather_data["sys"]["country"]
    weather_desc = weather_data["weather"][0]["description"].capitalize()
    temp = weather_data["main"]["temp"]
    feels_like = weather_data["main"]["feels_like"]
    humidity = weather_data["main"]["humidity"]
    wind_speed = weather_data["wind"]["speed"]
    
    # Convert timestamp to readable time
    sunrise_time = datetime.fromtimestamp(weather_data["sys"]["sunrise"]).strftime("%H:%M")
    sunset_time = datetime.fromtimestamp(weather_data["sys"]["sunset"]).strftime("%H:%M")
    
    output = {
    'location': city,
    'condition': weather_desc,
    'temperature': temp,
    'feelsLike': feels_like,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'sunrise': sunrise_time,
    'sunset': sunset_time
    }
    
    return output


def format_forecast(forecast_data: Dict[str, Any], days: int = 3) -> str:
    """
    Format forecast data into a readable string.
    
    Args:
        forecast_data: Forecast data dictionary from the API
        days: Number of days to display (default: 3)
        
    Returns:
        Formatted forecast information string
    """
    city = forecast_data["city"]["name"]
    country = forecast_data["city"]["country"]
    
    # Group forecasts by day
    forecasts_by_day = {}
    for item in forecast_data["list"]:
        day = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
        
        if day not in forecasts_by_day:
            forecasts_by_day[day] = []
        
        forecasts_by_day[day].append(item)
    
    # Format the output
    output = f"\nWeather Forecast for {city}, {country}:\n"
    
    # Get the keys (days) and sort them
    sorted_days = sorted(forecasts_by_day.keys())
    
    # Limit to the requested number of days
    for day in sorted_days[:days]:
        date_obj = datetime.strptime(day, "%Y-%m-%d")
        day_str = date_obj.strftime("%A, %b %d")
        
        output += f"\n{day_str}:\n{'=' * (len(day_str) + 1)}\n"
        
        for item in forecasts_by_day[day]:
            time = datetime.fromtimestamp(item["dt"]).strftime("%H:%M")
            temp = item["main"]["temp"]
            weather_desc = item["weather"][0]["description"].capitalize()
            
            output += f"{time}: {temp}°F - {weather_desc}\n"
    
    return output


def main():
    from dotenv import load_dotenv  
    load_dotenv()
    import os

    # AWS Credentials
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    parser = argparse.ArgumentParser(description="Get weather information for a city or zip code")
    
    parser.add_argument("location", help="City name or zip code")
    parser.add_argument("--zip", action="store_true", help="Indicate that the location is a zip code")
    parser.add_argument("--country", default="us", help="Two-letter country code (default: us)")
    parser.add_argument("--api-key", required=False, help="Your OpenWeatherMap API key")
    parser.add_argument("--forecast", action="store_true", help="Show forecast")
    parser.add_argument("--days", type=int, default=3, help="Number of days for forecast (default: 3)")
    
    args = parser.parse_args()
    
    try:
        weather_api = WeatherAPI(OPENWEATHER_API_KEY) #args.api_key)
        
        # Get and display current weather
        current_weather = weather_api.get_current_weather(args.location, args.zip, args.country)
        print(format_current_weather(current_weather))
        
        # Get and display forecast if requested
        if args.forecast:
            forecast = weather_api.get_forecast(args.location, args.zip, args.country, args.days)
            print(format_forecast(forecast, args.days))
            
    except Exception as e:
        print(f"Error: {str(e)}")


#if __name__ == "__main__":
#    main()