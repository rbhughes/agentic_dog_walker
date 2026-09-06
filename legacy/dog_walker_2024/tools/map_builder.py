"""Build Folium maps programmatically for Streamlit display."""

import folium
import requests
from typing import Any


def get_walking_route_geometry(
    start: list[float], end: list[float], api_key: str
) -> list[list[float]]:
    """
    Get the actual walking route geometry between two points from OpenRouteService.

    Args:
        start: [lat, lon] of start point
        end: [lat, lon] of end point
        api_key: OpenRouteService API key

    Returns:
        List of [lat, lon] coordinates along the route
    """
    if not api_key:
        # Fallback to straight line if no API key
        return [start, end]

    # OpenRouteService expects [lon, lat] format
    coordinates = [[start[1], start[0]], [end[1], end[0]]]

    headers = {
        "Authorization": api_key,
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


def build_route_map(
    locations: list[dict[str, Any]],
    route_sequence: list[int],
    api_key: str,
) -> folium.Map:
    """
    Build a Folium map with the optimized route.

    Args:
        locations: List of location dicts with latitude, longitude, pet_name, address, duration
        route_sequence: List of indices showing the route order
        api_key: OpenRouteService API key for route geometry

    Returns:
        Folium Map object
    """
    if not locations:
        # Return empty map centered on default location
        return folium.Map(location=[41.8781, -87.6298], zoom_start=13)

    # Determine map center
    map_center = [locations[0]["latitude"], locations[0]["longitude"]]

    # Create map
    m = folium.Map(location=map_center, zoom_start=13)

    if route_sequence:
        # Draw route segments between consecutive locations
        ordered_locations = [locations[i] for i in route_sequence]

        for i in range(len(ordered_locations) - 1):
            start_loc = ordered_locations[i]
            end_loc = ordered_locations[i + 1]

            start = [start_loc["latitude"], start_loc["longitude"]]
            end = [end_loc["latitude"], end_loc["longitude"]]

            # Get the actual walking route geometry
            route_coords = get_walking_route_geometry(start, end, api_key)

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
                popup=f"{segment_label}: {start_loc.get('pet_name', 'Unknown')} → {end_loc.get('pet_name', 'Unknown')}",
                dash_array="10, 10" if is_return_home else None,
            ).add_to(m)

        # Add numbered markers based on route sequence order
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
            address = loc.get("address", "Unknown")

            popup_parts = [f"Stop {stop_counter}: {pet_name or address}"]
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

    return m
