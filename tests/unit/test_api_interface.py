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


def test_system_message_included(mock_openai_client, api_interface):
    """Test that the system message is included in the API call."""
    mock_client, set_mock_response, expected_system_message = mock_openai_client

    # Set a valid mock response
    set_mock_response('{"key": "value"}')

    query = "Mock query"
    api_interface.send_query(query)

    # Verify that the system message is included in the API call
    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=[
            {"role": "system", "content": expected_system_message},
            {"role": "user", "content": "Mock query"},
        ],
        temperature=0.7,
    )


def test_send_query_success(mock_openai_client, api_interface):
    """Test successful query to the API."""
    mock_client, set_mock_response, _ = mock_openai_client

    # Set a valid mock response
    set_mock_response('{"key": "value"}')

    query = "Mock query"
    response = api_interface.send_query(query)

    # Assert that the response matches the mock
    assert response == '{"key": "value"}'
    mock_client.chat.completions.create.assert_called_once()


def test_send_query_invalid_json(mock_openai_client, api_interface):
    """Test retry logic for invalid JSON responses."""
    mock_client, set_mock_response, _ = mock_openai_client

    # Set an invalid JSON mock response
    set_mock_response('{"key": "value"')  # Invalid JSON (missing closing brace)

    query = "Mock query"

    # Expect a ValueError due to retries exhausting on invalid JSON
    with pytest.raises(ValueError, match="Invalid JSON response after retries"):
        api_interface.send_query(query)

    # Assert the API call was retried the correct number of times
    assert mock_client.chat.completions.create.call_count == 3


def test_send_query_retries_on_error(mock_openai_client, api_interface):
    """Test retry logic for API errors."""
    mock_client, _, _ = mock_openai_client

    # Simulate API error with a side effect
    mock_client.chat.completions.create.side_effect = Exception("Mock API error")

    query = "Mock query"

    # Expect a RuntimeError after retries are exhausted
    with pytest.raises(
        RuntimeError, match="API query failed after 3 retries: Mock API error"
    ):
        api_interface.send_query(query)

    # Assert that the API call was retried the correct number of times
    assert mock_client.chat.completions.create.call_count == 3


def test_send_query_retries_on_invalid_json(mock_openai_client, api_interface):
    """Test retry logic for invalid JSON responses."""
    mock_client, set_mock_response, _ = mock_openai_client

    # Set an invalid JSON mock response
    set_mock_response('{"key": "value"')  # Invalid JSON (missing closing brace)

    query = "Mock query"

    # Expect a ValueError due to retries exhausting on invalid JSON
    with pytest.raises(ValueError, match="Invalid JSON response after retries"):
        api_interface.send_query(query)

    # Assert the API call was retried the correct number of times
    assert mock_client.chat.completions.create.call_count == 3
