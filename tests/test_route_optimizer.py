"""Tests for route optimizer tool."""

import json
from unittest.mock import Mock, patch

import pytest

from dog_walker.tools.route_optimizer import optimize_dog_walking_route


@pytest.fixture
def sample_visits():
    """Sample visit data."""
    return {
        "visits": [
            {
                "pet_name": "Max",
                "coordinates": [41.8781, -87.6298],
                "duration": 30,
            },
            {
                "pet_name": "Bella",
                "coordinates": [41.8969, -87.6232],
                "duration": 20,
            },
            {
                "pet_name": "Charlie",
                "coordinates": [41.9214, -87.6551],
                "duration": 25,
            },
        ]
    }


def test_optimize_route_basic(sample_visits):
    """Test basic route optimization."""
    # Mock distance matrix API
    mock_response = Mock()
    mock_response.json.return_value = {
        "distances": [
            [0, 1000, 2000],
            [1000, 0, 1500],
            [2000, 1500, 0],
        ]
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        result = optimize_dog_walking_route(json.dumps(sample_visits))
        data = json.loads(result)

        assert "optimized_sequence" in data
        assert "visit_order" in data
        assert "locations_for_map" in data
        assert "total_distance_meters" in data
        assert "estimated_time_hours" in data

        # Should have 3 visits + return home
        assert len(data["optimized_sequence"]) == 4
        # First and last should be same (return home)
        assert data["optimized_sequence"][0] == data["optimized_sequence"][-1]


def test_optimize_route_visit_order(sample_visits):
    """Test that visit order is correctly formatted."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "distances": [[0, 1000, 2000], [1000, 0, 1500], [2000, 1500, 0]]
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        result = optimize_dog_walking_route(json.dumps(sample_visits))
        data = json.loads(result)

        # Check visit_order structure
        assert len(data["visit_order"]) == 3
        for visit in data["visit_order"]:
            assert "pet_name" in visit
            assert "address" in visit
            assert "duration_minutes" in visit


def test_optimize_route_locations_for_map(sample_visits):
    """Test that locations_for_map is correctly formatted."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "distances": [[0, 1000, 2000], [1000, 0, 1500], [2000, 1500, 0]]
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        result = optimize_dog_walking_route(json.dumps(sample_visits))
        data = json.loads(result)

        # Check locations_for_map structure
        assert len(data["locations_for_map"]) == 3
        for loc in data["locations_for_map"]:
            assert "latitude" in loc
            assert "longitude" in loc
            assert "pet_name" in loc
            assert "duration" in loc


def test_optimize_route_with_backticks():
    """Test handling of backtick-wrapped JSON."""
    visits = {
        "visits": [{"pet_name": "Max", "coordinates": [41.88, -87.63], "duration": 30}]
    }

    mock_response = Mock()
    mock_response.json.return_value = {"distances": [[0]]}
    mock_response.raise_for_status = Mock()

    with patch("requests.post", return_value=mock_response):
        # Wrap in backticks like LangChain might
        result = optimize_dog_walking_route(f"```{json.dumps(visits)}```")
        data = json.loads(result)

        assert "optimized_sequence" in data


def test_optimize_route_missing_visits_key():
    """Test error handling for missing visits key."""
    result = optimize_dog_walking_route('{"no_visits": []}')

    assert "failed" in result.lower()


def test_optimize_route_invalid_json():
    """Test error handling for invalid JSON."""
    result = optimize_dog_walking_route("{invalid json}")

    assert "failed" in result.lower()


def test_optimize_route_api_failure():
    """Test handling of API failures (falls back to haversine)."""
    visits = {
        "visits": [
            {"pet_name": "Max", "coordinates": [41.8781, -87.6298], "duration": 30},
            {"pet_name": "Bella", "coordinates": [41.8969, -87.6232], "duration": 20},
        ]
    }

    with patch("requests.post", side_effect=Exception("API Error")):
        result = optimize_dog_walking_route(json.dumps(visits))
        data = json.loads(result)

        # Should still work with haversine fallback
        assert "optimized_sequence" in data
        assert "total_distance_meters" in data
