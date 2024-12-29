import pytest
from openai_json.api_interface import APIInterface, AsyncAPIInterface
from openai import RateLimitError
from unittest.mock import MagicMock


@pytest.fixture
def api_interface():
    """Fixture for APIInterface instance."""
    return APIInterface(
        api_key="mock-api-key",
        model="gpt-4",
        retries=3,
        system_message="Respond in valid JSON format.",
        temperature=0.7,
    )


@pytest.fixture
def async_api_interface():
    """Fixture for AsyncAPIInterface instance."""
    return AsyncAPIInterface(
        api_key="mock-api-key",
        model="gpt-4",
        retries=3,
        system_message="Respond in valid JSON format.",
        temperature=0.7,
    )


def test_system_message_included(mock_openai_client, api_interface):
    """Test that the system message is included in the API call."""
    sync_mock_client, _, set_mock_response, expected_system_message = mock_openai_client

    # Set a valid mock response
    set_mock_response('{"key": "value"}')

    query = "Mock query"
    api_interface.send_query(query)

    # Verify that the system message is included in the API call
    sync_mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=[
            {"role": "system", "content": expected_system_message},
            {"role": "user", "content": "Mock query"},
        ],
        temperature=0.7,
    )


def test_send_query_success(mock_openai_client, api_interface):
    """Test successful query to the API."""
    sync_mock_client, _, set_mock_response, _ = mock_openai_client

    # Set a valid mock response
    set_mock_response('{"key": "value"}')

    query = "Mock query"
    response = api_interface.send_query(query)

    # Assert that the response matches the mock
    assert response == '{"key": "value"}'
    sync_mock_client.chat.completions.create.assert_called_once()


def test_send_query_invalid_json(mock_openai_client, api_interface):
    """Test retry logic for invalid JSON responses."""
    sync_mock_client, _, set_mock_response, _ = mock_openai_client

    # Mock an invalid JSON response
    set_mock_response('{"key": "value"')  # Missing closing brace

    query = "Mock query"

    # Expect a RuntimeError due to retries exhausting on invalid JSON
    with pytest.raises(
        RuntimeError, match="API query failed after retries: Invalid JSON response"
    ):
        api_interface.send_query(query)

    # Verify the retry count
    assert sync_mock_client.chat.completions.create.call_count == 3


from openai import RateLimitError


def test_send_query_retries_on_error(mock_openai_client, api_interface):
    """Test retry logic for API errors."""
    sync_mock_client, _, _, _ = mock_openai_client

    # Create a mock response with required attributes
    mock_response = MagicMock()
    mock_response.request = MagicMock()  # Add the `request` attribute
    mock_body = "Rate limit exceeded"

    # Simulate a retryable API error
    sync_mock_client.chat.completions.create.side_effect = RateLimitError(
        "Rate limit exceeded", response=mock_response, body=mock_body
    )

    query = "Mock query"

    # Expect a RuntimeError after retries are exhausted
    with pytest.raises(
        RuntimeError, match="Sync API query failed after retries: Rate limit exceeded"
    ):
        api_interface.send_query(query)

    # Verify the retry count
    assert sync_mock_client.chat.completions.create.call_count == 3


def test_send_query_retries_on_invalid_json(mock_openai_client, api_interface):
    """Test retry logic for invalid JSON responses."""
    sync_mock_client, _, set_mock_response, _ = mock_openai_client

    # Mock an invalid JSON response
    set_mock_response('{"key": "value"')  # Missing closing brace

    query = "Mock query"

    # Expect a RuntimeError due to retries exhausting on invalid JSON
    with pytest.raises(
        RuntimeError, match="Sync API query failed after retries: Invalid JSON response"
    ):
        api_interface.send_query(query)

    # Verify that the API call was retried the correct number of times
    assert sync_mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_async_system_message_included(mock_openai_client, async_api_interface):
    """Test that the system message is included in the async API call."""
    _, async_mock_client, set_mock_response, expected_system_message = (
        mock_openai_client
    )

    # Set a valid mock response
    set_mock_response('{"key": "value"}')

    query = "Mock query"
    await async_api_interface.send_query(query)

    # Verify that the system message is included in the API call
    async_mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=[
            {"role": "system", "content": expected_system_message},
            {"role": "user", "content": "Mock query"},
        ],
        temperature=0.7,
    )


@pytest.mark.asyncio
async def test_async_send_query_success(mock_openai_client, async_api_interface):
    """Test successful async query to the API."""
    _, async_mock_client, set_mock_response, _ = mock_openai_client

    # Set a valid mock response
    set_mock_response('{"key": "value"}')

    query = "Mock query"
    response = await async_api_interface.send_query(query)

    # Assert that the response matches the mock
    assert response == '{"key": "value"}'
    async_mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_async_send_query_invalid_json(mock_openai_client, async_api_interface):
    """Test retry logic for invalid JSON responses in async API."""
    _, async_mock_client, set_mock_response, _ = mock_openai_client

    # Mock an invalid JSON response
    set_mock_response('{"key": "value"')  # Missing closing brace

    query = "Mock query"

    # Expect a RuntimeError due to retries exhausting on invalid JSON
    with pytest.raises(
        RuntimeError,
        match="Async API query failed after retries: Invalid JSON response",
    ):
        await async_api_interface.send_query(query)

    # Verify the retry count
    assert async_mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_async_send_query_retries_on_error(
    mock_openai_client, async_api_interface
):
    """Test retry logic for API errors in async API."""
    _, async_mock_client, _, _ = mock_openai_client

    # Create a mock response with required attributes
    mock_response = MagicMock()
    mock_response.request = MagicMock()  # Add the `request` attribute
    mock_body = "Rate limit exceeded"

    # Simulate a retryable API error
    async_mock_client.chat.completions.create.side_effect = RateLimitError(
        "Rate limit exceeded", response=mock_response, body=mock_body
    )

    query = "Mock query"

    # Expect a RuntimeError after retries are exhausted
    with pytest.raises(
        RuntimeError, match="Async API query failed after retries: Rate limit exceeded"
    ):
        await async_api_interface.send_query(query)

    # Verify the retry count
    assert async_mock_client.chat.completions.create.call_count == 3
