import ast
import json

import requests
from langchain.tools import Tool


def geocode_addresses(addresses_str: str) -> str:
    """
    Convert addresses to coordinates using Nominatim API.

    Args:
        addresses_str: JSON array of addresses like ["addr1", "addr2"]

    Returns:
        JSON array with lat/lon for each address
    """
    # Parse input - handle both JSON [""] and Python [''] formats
    addresses_str = addresses_str.strip().strip("'\"")
    try:
        addresses = json.loads(addresses_str)
    except json.JSONDecodeError:
        try:
            addresses = ast.literal_eval(addresses_str)
        except (ValueError, SyntaxError):
            return json.dumps([{"address": addresses_str, "error": "Invalid format"}])

    # Ensure it's a list
    if not isinstance(addresses, list):
        addresses = [addresses]

    results = []
    for address in addresses:
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": address, "format": "json", "limit": 1},
                headers={"User-Agent": "agentic-dog-walker/0.1.0"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data:
                results.append(
                    {
                        "address": address,
                        "latitude": float(data[0]["lat"]),
                        "longitude": float(data[0]["lon"]),
                        "name": data[0].get("display_name", address),
                    }
                )
            else:
                results.append({"address": address, "error": "Not found"})
        except Exception as e:
            results.append({"address": address, "error": str(e)})

    return json.dumps(results)


geocoding_tool = Tool(
    name="geocode_addresses",
    description='Convert addresses to coordinates. Input: JSON array like ["123 Main St, Chicago, IL", "456 Oak Ave, Boston, MA"]',
    func=geocode_addresses,
)
