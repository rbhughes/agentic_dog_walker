import requests
from langchain.tools import Tool


def check_weather_impact(coordinates_and_time: str) -> str:
    """
    Check weather conditions for dog walking.

    Args:
        coordinates_and_time: "lat,lon,YYYY-MM-DD" format

    Returns:
        Weather summary and recommendations
    """
    try:
        lat, lon, date = coordinates_and_time.split(",")

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": date,
                "end_date": date,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        hourly = data.get("hourly", {})

        # Calculate averages
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        wind = hourly.get("wind_speed_10m", [])

        avg_temp = sum(temps) / len(temps) if temps else 0
        total_precip = sum(precip) if precip else 0
        max_wind = max(wind) if wind else 0

        # Generate recommendations
        recommendations = []
        if total_precip > 1.0:
            recommendations.append("Rainy - bring rain gear")
        if avg_temp > 30:
            recommendations.append("Hot - bring extra water")
        if avg_temp < 0:
            recommendations.append("Freezing - use paw protection")
        if max_wind > 30:
            recommendations.append("Windy - avoid open areas")

        return f"Weather: Temp {avg_temp:.1f}°C, Precip {total_precip:.1f}mm, Wind {max_wind:.1f}km/h. Recommendations: {'; '.join(recommendations) if recommendations else 'Good conditions for walking'}"

    except Exception as e:
        return f"Weather check failed: {str(e)}"


weather_tool = Tool(
    name="check_weather",
    description="Check weather for coordinates and date. Input: 'lat,lon,YYYY-MM-DD'",
    func=check_weather_impact,
)
