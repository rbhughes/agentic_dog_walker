import json
import requests
from typing import Any
import folium
from langchain.tools import Tool
from dog_walker.utils.config import OUTPUT_DIR, OPENROUTESERVICE_API_KEY


def get_walking_route_geometry(
    start: list[float], end: list[float]
) -> list[list[float]]:
    """
    Get the actual walking route geometry between two points from OpenRouteService.

    Args:
        start: [lat, lon] of start point
        end: [lat, lon] of end point

    Returns:
        List of [lat, lon] coordinates along the route
    """
    if not OPENROUTESERVICE_API_KEY:
        # Fallback to straight line if no API key
        return [start, end]

    # OpenRouteService expects [lon, lat] format
    coordinates = [[start[1], start[0]], [end[1], end[0]]]

    headers = {
        "Authorization": OPENROUTESERVICE_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "coordinates": coordinates,
    }

    try:
        response = requests.post(
            "https://api.openrouteservice.org/v2/directions/foot-walking/geojson",
            headers=headers,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        # Extract geometry from GeoJSON format - it's in [lon, lat] format
        geometry = data["features"][0]["geometry"]["coordinates"]

        # Convert to [lat, lon] format
        route_coords = [[coord[1], coord[0]] for coord in geometry]

        return route_coords

    except Exception as e:
        print(f"Warning: Failed to get route geometry ({str(e)}), using straight line")
        return [start, end]


def create_route_map(route_data_str: str) -> str:
    """
    Create an interactive HTML map of the dog walking route with street-following paths.

    Args:
        route_data_str: JSON string containing:
            - locations: List of dicts with address, latitude, longitude, name,
              optional pet_name, optional duration
            - route_sequence: Optional list of indices for the optimized route order
            - center: Optional [lat, lon] for map center (defaults to first location)

    Returns:
        Path to the generated HTML map file
    """
    try:
        route_data: dict[str, Any] = json.loads(route_data_str)

        # Prefer locations_for_map if provided (already in route order from optimizer)
        if "locations_for_map" in route_data:
            locations: list[dict[str, Any]] = route_data["locations_for_map"]
            # Create sequential route_sequence since locations are already ordered
            route_sequence: list[int] | None = list(range(len(locations))) + [0]
        else:
            # Fallback: use provided locations and route_sequence, but DON'T trust them to match
            # Instead, just render locations in the order provided without route optimization
            locations: list[dict[str, Any]] = route_data.get("locations", [])
            route_sequence: list[int] | None = (
                None  # Ignore the route_sequence if no locations_for_map
            )

        center: list[float] | None = route_data.get("center")

        if not locations:
            return "Error: No locations provided"

        # Determine map center
        if center:
            map_center = center
        else:
            map_center = [locations[0]["latitude"], locations[0]["longitude"]]

        # Create map
        m = folium.Map(location=map_center, zoom_start=13)

        # Add markers for each location
        if route_sequence:
            # If we have a route sequence, draw the route following streets
            ordered_locations = [locations[i] for i in route_sequence]

            # Draw route segments between consecutive locations
            for i in range(len(ordered_locations) - 1):
                start_loc = ordered_locations[i]
                end_loc = ordered_locations[i + 1]

                start = [start_loc["latitude"], start_loc["longitude"]]
                end = [end_loc["latitude"], end_loc["longitude"]]

                # Get the actual walking route geometry
                route_coords = get_walking_route_geometry(start, end)

                # Check if this is the return home segment
                is_return_home = (
                    i == len(ordered_locations) - 2
                    and route_sequence[-1] == route_sequence[0]
                )
                segment_label = "Return Home" if is_return_home else f"Segment {i + 1}"

                folium.PolyLine(
                    route_coords,
                    color="green" if is_return_home else "blue",
                    weight=4,
                    opacity=0.6 if is_return_home else 0.8,
                    popup=f"{segment_label}: {start_loc.get('name', 'Unknown')} → {end_loc.get('name', 'Unknown')}",
                    dashArray="10, 10" if is_return_home else None,
                ).add_to(m)

            # Add numbered markers based on route sequence order
            # Track which locations we've already added markers for
            location_to_stop_number = {}
            stop_counter = 1

            for idx in route_sequence:
                # Skip if we've already added a marker for this location
                if idx in location_to_stop_number:
                    continue

                location_to_stop_number[idx] = stop_counter
                loc = locations[idx]

                # Build popup text with pet name and duration if available
                pet_name = loc.get("pet_name", "")
                duration = loc.get("duration", "")
                location_name = loc.get("name", loc.get("address", "Unknown"))

                popup_parts = [f"Stop {stop_counter}: {location_name}"]
                if pet_name:
                    popup_parts.append(f"Pet: {pet_name}")
                if duration:
                    popup_parts.append(f"Duration: {duration} min")

                popup_text = "<br>".join(popup_parts)

                # Build tooltip
                tooltip_parts = [f"Stop {stop_counter}"]
                if pet_name:
                    tooltip_parts.append(pet_name)
                tooltip_text = " - ".join(tooltip_parts)

                folium.Marker(
                    location=[loc["latitude"], loc["longitude"]],
                    popup=popup_text,
                    tooltip=tooltip_text,
                    icon=folium.Icon(
                        color="green" if stop_counter == 1 else "blue",
                        icon="info-sign",
                    ),
                ).add_to(m)

                stop_counter += 1
        else:
            # No route sequence, just add markers
            for idx, loc in enumerate(locations):
                folium.Marker(
                    location=[loc["latitude"], loc["longitude"]],
                    popup=f"{loc.get('name', loc.get('address', 'Unknown'))}",
                    tooltip=f"Location {idx + 1}",
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(m)

        # Save map
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"route_map_{timestamp}.html"
        filepath = OUTPUT_DIR / filename

        m.save(str(filepath))

        return f"Map saved to: {filepath}"

    except Exception as e:
        return f"Map creation failed: {str(e)}"


# Create LangChain tool
mapping_tool = Tool(
    name="create_route_map",
    description='Create an interactive HTML map showing the optimized dog walking route with street-following paths. Input must be valid JSON with "locations" array (each with latitude, longitude, name, address, pet_name, duration) and "route_sequence" array (indices from optimize_route result). Example: {"locations": [{"latitude": 41.88, "longitude": -87.63, "name": "Max", "address": "123 Main St", "pet_name": "Max", "duration": 30}], "route_sequence": [0, 1, 2, 0]}',
    func=create_route_map,
)
