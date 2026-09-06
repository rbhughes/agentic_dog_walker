import json
from datetime import datetime

import folium
import requests
from langchain.tools import Tool

from dog_walker.utils.config import OUTPUT_DIR, OPENROUTESERVICE_API_KEY


def get_walking_route(start: list[float], end: list[float]) -> list[list[float]]:
    """
    Get walking route geometry between two points from OpenRouteService.

    Args:
        start: [lat, lon] start point
        end: [lat, lon] end point

    Returns:
        List of [lat, lon] coordinates along route
    """
    if not OPENROUTESERVICE_API_KEY:
        return [start, end]  # Fallback to straight line

    try:
        response = requests.post(
            "https://api.openrouteservice.org/v2/directions/foot-walking/geojson",
            headers={
                "Authorization": OPENROUTESERVICE_API_KEY,
                "Content-Type": "application/json",
            },
            json={"coordinates": [[start[1], start[0]], [end[1], end[0]]]},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # Convert from [lon, lat] to [lat, lon]
        geometry = data["features"][0]["geometry"]["coordinates"]
        return [[coord[1], coord[0]] for coord in geometry]
    except Exception:
        return [start, end]


def create_route_map(route_data_str: str) -> str:
    """
    Create interactive HTML map of optimized dog walking route.

    Args:
        route_data_str: JSON from optimize_route with locations_for_map

    Returns:
        Path to generated HTML map file
    """
    try:
        # Clean input
        route_data_str = route_data_str.strip()
        json_start = route_data_str.find("{")
        if json_start > 0:
            route_data_str = route_data_str[json_start:]
        route_data_str = route_data_str.strip("`'\"")

        data = json.loads(route_data_str)

        # Use locations_for_map (already in route order from optimizer)
        locations = data.get("locations_for_map", [])
        if not locations:
            return "Error: No locations_for_map found in input"

        # Create map centered on first location
        map_center = [locations[0]["latitude"], locations[0]["longitude"]]
        m = folium.Map(location=map_center, zoom_start=13)

        # Draw route segments between consecutive locations
        for i in range(len(locations)):
            current = locations[i]
            next_loc = locations[(i + 1) % len(locations)]  # Loop back to start

            # Get walking route geometry
            start = [current["latitude"], current["longitude"]]
            end = [next_loc["latitude"], next_loc["longitude"]]
            route_coords = get_walking_route(start, end)

            # Draw route (dashed for return home)
            is_return = i == len(locations) - 1
            folium.PolyLine(
                route_coords,
                color="green" if is_return else "blue",
                weight=4,
                opacity=0.6 if is_return else 0.8,
                dashArray="10, 10" if is_return else None,
            ).add_to(m)

        # Add numbered markers
        for i, loc in enumerate(locations, 1):
            pet_name = loc.get("pet_name", "Unknown")
            duration = loc.get("duration", "")

            popup_parts = [f"Stop {i}: {pet_name}"]
            if duration:
                popup_parts.append(f"Duration: {duration} min")

            folium.Marker(
                location=[loc["latitude"], loc["longitude"]],
                popup="<br>".join(popup_parts),
                tooltip=f"Stop {i} - {pet_name}",
                icon=folium.Icon(color="green" if i == 1 else "blue", icon="info-sign"),
            ).add_to(m)

        # Save map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = OUTPUT_DIR / f"route_map_{timestamp}.html"
        m.save(str(filepath))

        return f"Map saved to: {filepath}"

    except Exception as e:
        return f"Map creation failed: {str(e)}"


mapping_tool = Tool(
    name="create_route_map",
    description="Create interactive HTML map from optimize_route output. Input: exact JSON from optimize_route (must have locations_for_map)",
    func=create_route_map,
)
