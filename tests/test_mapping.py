"""Tests for mapping tool."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from dog_walker.tools.mapping import create_route_map


@pytest.fixture
def sample_route_data():
    """Sample optimized route data."""
    return {
        "optimized_sequence": [0, 1, 2, 0],
        "locations_for_map": [
            {
                "latitude": 41.8781,
                "longitude": -87.6298,
                "pet_name": "Max",
                "address": "123 Main St",
                "duration": 30,
            },
            {
                "latitude": 41.8969,
                "longitude": -87.6232,
                "pet_name": "Bella",
                "address": "456 Oak Ave",
                "duration": 20,
            },
            {
                "latitude": 41.9214,
                "longitude": -87.6551,
                "pet_name": "Charlie",
                "address": "789 Elm St",
                "duration": 25,
            },
        ],
    }


def test_create_map_success(sample_route_data):
    """Test successful map creation."""
    # Mock route geometry API
    mock_response = Mock()
    mock_response.json.return_value = {
        "features": [
            {
                "geometry": {
                    "coordinates": [
                        [-87.6298, 41.8781],
                        [-87.6290, 41.8785],
                        [-87.6232, 41.8969],
                    ]
                }
            }
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        with patch("folium.Map.save") as mock_save:
            result = create_route_map(json.dumps(sample_route_data))

            assert "Map saved to:" in result
            assert "route_map_" in result
            assert ".html" in result
            # Verify save was called
            mock_save.assert_called_once()


def test_create_map_with_text_prefix(sample_route_data):
    """Test map creation with natural language text before JSON."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "features": [{"geometry": {"coordinates": [[-87.6298, 41.8781]]}}]
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        with patch("folium.Map.save"):
            # Add text before JSON
            input_str = f"Here is the route: {json.dumps(sample_route_data)}"
            result = create_route_map(input_str)

            assert "Map saved to:" in result


def test_create_map_with_backticks(sample_route_data):
    """Test map creation with backtick-wrapped JSON."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "features": [{"geometry": {"coordinates": [[-87.6298, 41.8781]]}}]
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        with patch("folium.Map.save"):
            # Wrap in backticks
            input_str = f"```{json.dumps(sample_route_data)}```"
            result = create_route_map(input_str)

            assert "Map saved to:" in result


def test_create_map_no_locations():
    """Test error handling when no locations_for_map."""
    result = create_route_map('{"optimized_sequence": [0, 1, 0]}')

    assert "Error" in result
    assert "No locations_for_map" in result


def test_create_map_invalid_json():
    """Test error handling for invalid JSON."""
    result = create_route_map("{invalid json}")

    assert "Map creation failed" in result


def test_create_map_api_fallback(sample_route_data):
    """Test fallback to straight lines when route API fails."""
    with patch("requests.post", side_effect=Exception("API Error")):
        with patch("folium.Map.save"):
            result = create_route_map(json.dumps(sample_route_data))

            # Should still create map with straight lines
            assert "Map saved to:" in result


def test_create_map_without_api_key(sample_route_data):
    """Test map creation when no API key is configured."""
    with patch("dog_walker.tools.mapping.OPENROUTESERVICE_API_KEY", ""):
        with patch("folium.Map.save"):
            result = create_route_map(json.dumps(sample_route_data))

            # Should create map with straight lines
            assert "Map saved to:" in result
