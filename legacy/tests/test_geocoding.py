"""Tests for geocoding tool."""

import json
from unittest.mock import Mock, patch

import pytest

from dog_walker.tools.geocoding import geocode_addresses


def test_geocode_single_address() -> None:
    """Test geocoding a single address in JSON array format."""
    mock_response = Mock()
    mock_response.json.return_value = [
        {
            "lat": "41.8781",
            "lon": "-87.6298",
            "display_name": "Chicago, IL",
        }
    ]
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = geocode_addresses('["108 N State St, Chicago"]')
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["latitude"] == 41.8781
        assert data[0]["longitude"] == -87.6298
        assert "Chicago" in data[0]["name"]


def test_geocode_multiple_addresses() -> None:
    """Test geocoding multiple addresses."""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"lat": "41.8781", "lon": "-87.6298", "display_name": "Chicago, IL"}
    ]
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = geocode_addresses('["Address 1", "Address 2"]')
        data = json.loads(result)

        assert len(data) == 2


def test_geocode_python_list_format() -> None:
    """Test that Python list format ['addr'] is handled."""
    mock_response = Mock()
    mock_response.json.return_value = [
        {"lat": "41.8781", "lon": "-87.6298", "display_name": "Chicago, IL"}
    ]
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        # Python list format with single quotes
        result = geocode_addresses("['108 N State St, Chicago']")
        data = json.loads(result)

        assert len(data) == 1
        assert data[0]["latitude"] == 41.8781


def test_geocode_address_not_found() -> None:
    """Test handling of address not found."""
    mock_response = Mock()
    mock_response.json.return_value = []  # No results
    mock_response.raise_for_status = Mock()

    with patch("requests.get", return_value=mock_response):
        result = geocode_addresses('["Invalid Address"]')
        data = json.loads(result)

        assert len(data) == 1
        assert "error" in data[0]
        assert data[0]["error"] == "Not found"


def test_geocode_api_error() -> None:
    """Test handling of API errors."""
    with patch("requests.get", side_effect=Exception("API Error")):
        result = geocode_addresses('["Address"]')
        data = json.loads(result)

        assert len(data) == 1
        assert "error" in data[0]
        assert "API Error" in data[0]["error"]


def test_geocode_invalid_input() -> None:
    """Test handling of invalid input format."""
    result = geocode_addresses("{invalid json}")
    data = json.loads(result)

    assert len(data) == 1
    assert "error" in data[0]
    assert "Invalid format" in data[0]["error"]
