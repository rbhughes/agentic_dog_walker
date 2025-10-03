import requests
from langchain.tools import Tool


def check_weather_impact(coordinates_and_time: str) -> str:
    """
    Check weather conditions and assess impact on dog walking.

    Args:
        coordinates_and_time: "lat,lon,YYYY-MM-DD" format

    Returns:
        Weather assessment and recommendations
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
                "hourly": "temperature_2m,precipitation,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
        )

        data = response.json()
        hourly = data.get("hourly", {})

        # Analyze conditions
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        wind = hourly.get("wind_speed_10m", [])

        avg_temp = sum(temps) / len(temps) if temps else 0
        total_precip = sum(precip) if precip else 0
        max_wind = max(wind) if wind else 0

        # Generate recommendations
        recommendations = []
        if total_precip > 1.0:
            recommendations.append(
                "High precipitation - consider shorter routes or covered areas"
            )
        if avg_temp > 25:
            recommendations.append(
                "Hot weather - bring extra water and avoid midday walks"
            )
        if avg_temp < 0:
            recommendations.append(
                "Freezing conditions - shorter walks and paw protection needed"
            )
        if max_wind > 20:
            recommendations.append("Strong winds - avoid open areas")

        return f"Weather: Temp {avg_temp:.1f}°C, Precip {total_precip:.1f}mm, Wind {max_wind:.1f}km/h. Recommendations: {'; '.join(recommendations) if recommendations else 'Good conditions for walking'}"

    except Exception as e:
        return f"Weather check failed: {str(e)}"


weather_tool = Tool(
    name="check_weather",
    description="Check weather conditions for coordinates and date. Input format: 'lat,lon,YYYY-MM-DD'",
    func=check_weather_impact,
)
