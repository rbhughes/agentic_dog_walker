"""Tests for weather tool."""

from unittest.mock import Mock, patch

import pytest

from dog_walker.tools.weather import check_weather_impact


def test_weather_good_conditions():
    """Test weather check with good conditions."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "temperature_2m": [20, 21, 22, 23],
            "precipitation": [0, 0, 0, 0],
            "wind_speed_10m": [10, 12, 11, 9],
        }
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = check_weather_impact("41.88,-87.63,2025-10-03")

        assert "Weather:" in result
        assert "21.5°C" in result  # Average temp
        assert "0.0mm" in result  # No precipitation
        assert "Good conditions for walking" in result


def test_weather_rainy():
    """Test weather check with rain."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "temperature_2m": [15, 16, 15, 14],
            "precipitation": [0.5, 1.0, 0.8, 0.3],
            "wind_speed_10m": [15, 16, 14, 13],
        }
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = check_weather_impact("41.88,-87.63,2025-10-03")

        assert "Rainy - bring rain gear" in result
        assert "2.6mm" in result


def test_weather_hot():
    """Test weather check with hot conditions."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "temperature_2m": [32, 33, 34, 33],
            "precipitation": [0, 0, 0, 0],
            "wind_speed_10m": [10, 11, 10, 9],
        }
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = check_weather_impact("41.88,-87.63,2025-10-03")

        assert "Hot - bring extra water" in result


def test_weather_freezing():
    """Test weather check with freezing conditions."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "temperature_2m": [-2, -3, -1, 0],
            "precipitation": [0, 0, 0, 0],
            "wind_speed_10m": [10, 11, 10, 9],
        }
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = check_weather_impact("41.88,-87.63,2025-10-03")

        assert "Freezing - use paw protection" in result


def test_weather_windy():
    """Test weather check with windy conditions."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "temperature_2m": [20, 21, 22, 23],
            "precipitation": [0, 0, 0, 0],
            "wind_speed_10m": [35, 40, 38, 33],
        }
    }
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = check_weather_impact("41.88,-87.63,2025-10-03")

        assert "Windy - avoid open areas" in result
        assert "40.0km/h" in result


def test_weather_api_error():
    """Test handling of API errors."""
    with patch("requests.get", side_effect=Exception("API Error")):
        result = check_weather_impact("41.88,-87.63,2025-10-03")

        assert "Weather check failed" in result
        assert "API Error" in result


def test_weather_invalid_input():
    """Test handling of invalid input format."""
    result = check_weather_impact("invalid")

    assert "Weather check failed" in result
