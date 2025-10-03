#!/usr/bin/env python3

import sys
import json

sys.path.append("src")

from dog_walker.tools.geocoding import geocoding_tool
from dog_walker.tools.weather import weather_tool
from dog_walker.tools.mapping import mapping_tool
from dog_walker.tools.route_optimizer import route_optimizer_tool, Visit


def test_tools():
    """Test individual tools before building the agent."""

    # Test geocoding
    print("Testing geocoding...")
    result = geocoding_tool.run("108 N State St Chicago IL 60602")
    print(f"Geocoding result: {result}\n")

    # Test weather (using Chicago coordinates)
    print("Testing weather...")
    weather_result = weather_tool.run("41.8781,-87.6298,2025-10-03")
    print(f"Weather result: {weather_result}\n")

    # Test route optimizer with new visits format
    print("Testing route optimizer...")
    visits: list[Visit] = [
        {
            "pet_name": "Max",
            "address": "108 N State St, Chicago",
            "coordinates": [41.8781, -87.6298],
            "duration": 30,
        },
        {
            "pet_name": "Bella",
            "address": "2001 N Clark St, Chicago",
            "coordinates": [41.8969, -87.6232],
            "duration": 20,
        },
        {
            "pet_name": "Charlie",
            "address": "1060 W Addison St, Chicago",
            "coordinates": [41.9214, -87.6551],
            "duration": 25,
        },
        {
            "pet_name": "Luna",
            "address": "201 E Randolph St, Chicago",
            "coordinates": [41.8819, -87.6278],
            "duration": 15,
        },
    ]

    route_data = {"visits": visits}
    optimizer_result = route_optimizer_tool.run(json.dumps(route_data))
    print(f"Route optimizer result: {optimizer_result}\n")

    # Parse optimizer result to get the optimized sequence
    optimizer_result_data = json.loads(optimizer_result)
    optimized_sequence = optimizer_result_data.get(
        "optimized_sequence", list(range(len(visits)))
    )

    # Test mapping using locations_for_map from optimizer result
    print("Testing mapping with optimized route...")

    # The optimizer already provides locations_for_map in route order
    # Just pass the optimizer result directly to the mapping tool
    map_result = mapping_tool.run(optimizer_result)
    print(f"Mapping result: {map_result}\n")


if __name__ == "__main__":
    test_tools()
