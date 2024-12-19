from unittest.mock import patch
import pytest
from openai_json.api_interface import APIInterface


@pytest.fixture
def api_interface():
    return APIInterface("mock-api-key")


@patch("openai.ChatCompletion.create")
def test_send_query(mock_create, api_interface):
    mock_create.return_value = {
        "choices": [{"message": {"content": '{"key": "value"}'}}]
    }
    query = "Mock query"
    response = api_interface.send_query(query)
    assert response == '{"key": "value"}'


def test_parse_response_valid(api_interface):
    response = '{"name": "John", "age": 30}'
    parsed = api_interface.parse_response(response)
    assert parsed == {"name": "John", "age": 30}


def test_parse_response_invalid(api_interface):
    invalid_response = '{"name": "John", "age": 30'  # Missing closing brace
    with pytest.raises(ValueError, match="Invalid JSON response"):
        api_interface.parse_response(invalid_response)
