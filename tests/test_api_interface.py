from unittest.mock import patch, MagicMock
import pytest
from openai_json.api_interface import APIInterface


@pytest.fixture
def api_interface():
    """Fixture for APIInterface instance."""
    return APIInterface(
        api_key="mock-api-key",
        model="gpt-4",
        retries=3,
        system_message="Respond in valid JSON format.",
        temperature=0.7,
        schema=None,
    )


@patch("openai.OpenAI")
def test_send_query_success(mock_openai, api_interface):
    """Test successful query to the API."""
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock API response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"key": "value"}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    query = "Mock query"
    response = api_interface.send_query(query)

    assert response == '{"key": "value"}'
    mock_client.chat.completions.create.assert_called_once()


@patch("openai.OpenAI")
def test_send_query_invalid_json(mock_openai, api_interface):
    """Test handling of invalid JSON responses."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock invalid JSON response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"key": "value"'))  # Invalid JSON
    ]
    mock_client.chat.completions.create.return_value = mock_response

    query = "Mock query"
    with pytest.raises(ValueError, match="Invalid JSON response"):
        api_interface.send_query(query)

    mock_client.chat.completions.create.assert_called_once()


@patch("openai.OpenAI")
def test_send_query_retries_on_error(mock_openai, api_interface):
    """Test retry logic for API errors."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Simulate API error
    mock_client.chat.completions.create.side_effect = Exception("Mock API error")

    query = "Mock query"
    with pytest.raises(RuntimeError, match="API query failed after 3 retries"):
        api_interface.send_query(query)

    assert mock_client.chat.completions.create.call_count == 3


@patch("openai.OpenAI")
def test_send_query_retries_on_invalid_json(mock_openai, api_interface):
    """Test retry logic for invalid JSON responses."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock invalid JSON response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"key": "value"'))  # Invalid JSON
    ]
    mock_client.chat.completions.create.return_value = mock_response

    query = "Mock query"
    with pytest.raises(RuntimeError, match="API query failed after 3 retries"):
        api_interface.send_query(query)

    assert mock_client.chat.completions.create.call_count == 3