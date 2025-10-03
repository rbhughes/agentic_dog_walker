import json
import requests
from typing import Any
from ortools.constraint_solver import routing_enums_pb2, pywrapcp  # pyright: ignore[reportMissingTypeStubs]
import math
from langchain.tools import Tool
from dog_walker.utils.config import OPENROUTESERVICE_API_KEY, OPENROUTESERVICE_URL


def calculate_distance(
    coord1: tuple[float, float], coord2: tuple[float, float]
) -> float:
    """Calculate haversine distance between two coordinates (fallback method)."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    return 6371 * c  # Earth radius in km


def get_walking_distance_matrix(coordinates: list[list[float]]) -> list[list[int]]:
    """
    Get real walking distances using OpenRouteService API.

    Args:
        coordinates: List of [lat, lon] pairs

    Returns:
        Distance matrix in meters
    """
    if not OPENROUTESERVICE_API_KEY:
        raise ValueError("OpenRouteService API key not found in environment variables")

    # OpenRouteService expects [lon, lat] format
    locations = [[coord[1], coord[0]] for coord in coordinates]

    headers = {
        "Authorization": OPENROUTESERVICE_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "locations": locations,
        "metrics": ["distance"],
        "units": "m",
    }

    try:
        response = requests.post(
            OPENROUTESERVICE_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # Extract distance matrix
        distance_matrix: list[list[int]] = []
        for row in data["distances"]:
            distance_matrix.append([int(d) for d in row])

        return distance_matrix

    except Exception as e:
        # Fallback to haversine distance if API fails
        print(
            f"Warning: OpenRouteService API failed ({str(e)}), using haversine fallback"
        )
        num_locations = len(coordinates)
        distance_matrix = [[0] * num_locations for _ in range(num_locations)]

        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    distance_matrix[i][j] = int(
                        calculate_distance(
                            (coordinates[i][0], coordinates[i][1]),
                            (coordinates[j][0], coordinates[j][1]),
                        )
                        * 1000
                    )

        return distance_matrix


def optimize_dog_walking_route(route_data: str) -> str:
    """
    Optimize dog walking route using TSP with time windows and real walking distances.

    Args:
        route_data: JSON string with either:
            - visits: List of dicts with pet_name, address (optional), coordinates [lat,lon],
              duration (optional, defaults to 30), time_window (optional [start_hour, end_hour])
            OR legacy format:
            - coordinates, time_windows, durations

    Returns:
        Optimized route sequence and timing with visit details
    """
    try:
        data: dict[str, Any] = json.loads(route_data)

        # Check if using new visits format or legacy format
        if "visits" in data:
            visits: list[dict[str, Any]] = data["visits"]
            coordinates: list[list[float]] = [
                v.get("coordinates", [0, 0]) for v in visits
            ]
            durations: list[int] = [v.get("duration", 30) for v in visits]
            time_windows: list[list[float]] = [v.get("time_window", []) for v in visits]
            pet_names: list[str] = [
                v.get("pet_name", f"Pet {i + 1}") for i, v in enumerate(visits)
            ]
            addresses: list[str] = [v.get("address", "Unknown") for v in visits]
        else:
            # Legacy format
            coordinates: list[list[float]] = data["coordinates"]  # List of [lat, lon]
            time_windows: list[list[float]] = data.get(
                "time_windows", []
            )  # List of [start_hour, end_hour]
            durations: list[int] = data.get(
                "durations", [30] * len(coordinates)
            )  # Minutes per visit
            pet_names: list[str] = [f"Pet {i + 1}" for i in range(len(coordinates))]
            addresses: list[str] = data.get("addresses", ["Unknown"] * len(coordinates))

        # Get real walking distance matrix from OpenRouteService
        distance_matrix = get_walking_distance_matrix(coordinates)

        # Create routing model with return to start depot
        num_locations = len(coordinates)
        manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add time constraints if provided
        if time_windows:
            time_dimension = routing.AddDimension(
                transit_callback_index,
                30 * 60,  # 30 minutes slack
                12 * 60 * 60,  # 12 hours max
                False,
                "Time",
            )

            for location_idx, time_window in enumerate(time_windows):
                if len(time_window) == 2:
                    start_time = int(
                        time_window[0] * 60 * 60
                    )  # Convert hours to seconds
                    end_time = int(time_window[1] * 60 * 60)
                    index = manager.NodeToIndex(location_idx)
                    time_dimension.CumulVar(index).SetRange(start_time, end_time)

        # Solve
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            # Extract route including return to start
            route_sequence: list[int] = []
            index = routing.Start(0)
            total_distance = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_sequence.append(node_index)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)

            # Add return to start for complete circuit
            route_sequence.append(route_sequence[0])

            # Average walking speed: 5 km/h = 83.33 m/min
            walking_time_minutes = total_distance / 83.33
            total_time_hours = (walking_time_minutes + sum(durations)) / 60

            # Build detailed visit order with full location data for mapping
            visit_details = []
            locations_for_map = []

            for idx in route_sequence[:-1]:  # Exclude the duplicate return home
                visit_details.append(
                    {
                        "pet_name": pet_names[idx],
                        "address": addresses[idx]
                        if addresses[idx] != "Unknown"
                        else "",
                        "duration_minutes": durations[idx],
                    }
                )

                # Build location data for map in route order
                locations_for_map.append(
                    {
                        "latitude": coordinates[idx][0],
                        "longitude": coordinates[idx][1],
                        "pet_name": pet_names[idx],
                        "address": addresses[idx]
                        if addresses[idx] != "Unknown"
                        else "",
                        "duration": durations[idx],
                    }
                )

            return json.dumps(
                {
                    "optimized_sequence": route_sequence,
                    "visit_order": visit_details,
                    "locations_for_map": locations_for_map,
                    "total_distance_meters": total_distance,
                    "estimated_time_hours": round(total_time_hours, 2),
                    "walking_time_minutes": round(walking_time_minutes, 1),
                    "visit_time_minutes": sum(durations),
                    "uses_real_streets": True,
                }
            )
        else:
            return "No solution found"

    except Exception as e:
        return f"Route optimization failed: {str(e)}"


route_optimizer_tool = Tool(
    name="optimize_route",
    description='Optimize dog walking route using real street walking distances. Input must be valid JSON with "visits" array. Each visit needs: pet_name (string), coordinates (array [latitude, longitude]), duration (number in minutes). Example input: {"visits": [{"pet_name": "Max", "coordinates": [41.88, -87.63], "duration": 30}]}. The coordinates should come from geocode_addresses results.',
    func=optimize_dog_walking_route,
)
