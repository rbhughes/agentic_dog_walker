import requests
import json
from langchain.tools import Tool


def geocode_addresses(addresses_str: str) -> str:
    """
    Convert addresses to coordinates using Nominatim (OpenStreetMap) geocoding API.

    Args:
        addresses_str: JSON array string of addresses

    Returns:
        JSON string of coordinates
    """
    # Strip extra quotes that LangChain might add
    addresses_str = addresses_str.strip().strip("'\"")

    try:
        addresses = json.loads(addresses_str)
        if not isinstance(addresses, list):
            addresses = [addresses]
    except json.JSONDecodeError:
        # If it's not valid JSON, treat as single address
        addresses = [addresses_str]

    results = []

    for address in addresses:
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": address,
                    "format": "json",
                    "limit": 1,
                },
                headers={"User-Agent": "agentic-dog-walker/0.1.0"},
            )
            data = response.json()

            if data and len(data) > 0:
                result = data[0]
                results.append(
                    {
                        "address": address,
                        "latitude": float(result["lat"]),
                        "longitude": float(result["lon"]),
                        "name": result.get("display_name", address),
                    }
                )
            else:
                results.append({"address": address, "error": "Not found"})

        except Exception as e:
            results.append({"address": address, "error": str(e)})

    return str(results)


# Create LangChain tool
geocoding_tool = Tool(
    name="geocode_addresses",
    description='Convert street addresses to latitude/longitude coordinates. Input should be a JSON array of address strings, e.g., \'["123 Main St, Chicago, IL", "456 Oak Ave, Boston, MA"]\'',
    func=geocode_addresses,
)
