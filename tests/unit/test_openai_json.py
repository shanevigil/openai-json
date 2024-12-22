import pytest
from unittest.mock import patch
from openai_json.openai_json import OpenAI_JSON


@pytest.fixture
def expected_messages():
    """Fixture to construct expected messages with a system message."""

    def build(query, system_message="Respond in valid JSON format."):
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

    return build


def test_OpenAI_JSON_valid(mock_openai_client, expected_messages):
    """Integration test for OpenAI_JSON with valid inputs and mocked OpenAI client."""
    mock_client, set_mock_response, _ = mock_openai_client

    # Prepare the schema and query
    schema = {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    }
    query = "Generate a JSON object with name, age, and email."

    # Mock response from OpenAI API
    set_mock_response('{"name": "John", "age": 30, "email": "john@example.com"}')

    # Create OpenAI_JSON instance
    client = OpenAI_JSON(gpt_api_key="mock-api-key")
    client.api_client = mock_client

    # Expected messages sent to the OpenAI API
    expected_payload = expected_messages(query)

    # Run the handle_request method
    response = client.handle_request(query, schema)

    # Assertions for the response
    assert response == {
        "name": "John",
        "age": 30,
        "email": "john@example.com",
    }
    assert client.unmatched_data == []
    assert client.errors == []

    # Verify the messages sent to the OpenAI API
    # mock_client.chat_completions.create.assert_called_once_with(
    #     model="gpt-4",
    #     messages=expected_payload,
    #     temperature=0.7,
    # )


def test_OpenAI_JSON_with_unmatched_data(mock_openai_client, expected_messages):
    mock_client, set_mock_response, _ = mock_openai_client

    schema = {"First Name": {"type": "string"}, "Age": {"type": "integer"}}
    query = "Generate a JSON object with name and age."

    # Mock response with extra fields
    set_mock_response('{"first_name": "Alice", "age": 25, "extra": "unexpected"}')

    client = OpenAI_JSON(gpt_api_key="mock-api-key")
    client.api_client = mock_client

    expected_payload = expected_messages(query)

    response = client.handle_request(query, schema)

    assert response == {"First Name": "Alice", "Age": 25}
    assert client.unmatched_data == [{"extra": "unexpected"}]
    assert client.errors == []

    # mock_client.chat_completions.create.assert_called_once_with(
    #     model="gpt-4",
    #     messages=expected_payload,
    #     temperature=0.7,
    # )


def test_OpenAI_JSON_with_errors(mock_openai_client, expected_messages):
    """Test OpenAI_JSON with inputs that produce errors during heuristic processing."""
    mock_client, set_mock_response, _ = mock_openai_client

    # Prepare the schema and query
    schema = {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    }
    query = "Generate a JSON object with name, age, and email."

    # Mock response that has type mismatches
    set_mock_response(
        '{"name": "Alice", "age": "twenty-five", "email": "alice@someplace.com"}'
    )

    # Create OpenAI_JSON instance
    client = OpenAI_JSON(gpt_api_key="mock-api-key")
    client.api_client = mock_client

    # Run the handle_request method
    response = client.handle_request(query, schema)

    # Assertions for the response
    assert response == {
        "name": "Alice",
        "email": "alice@someplace.com",
    }  # Only valid data should be returned
    assert client.unmatched_data == []  # No unmatched keys since keys exist in response
    assert client.errors == [
        {"age": "twenty-five"},  # 'age' has a type error
    ]

    # Ensure the OpenAI client was called with the expected payload
    # expected_payload = expected_messages(query)
    # mock_client.chat_completions.create.assert_called_once_with(
    #     model="gpt-4",
    #     messages=expected_payload,
    #     temperature=0.7,
    # )


@pytest.mark.skip(reason="ML functionality not yet implemented")
def test_OpenAI_JSON_with_ml_processor(mock_openai_client, expected_messages):
    mock_client, set_mock_response, expected_system_message = mock_openai_client

    # Simulate a challenging response for the ML_processor to handle
    set_mock_response('{"Name": "John", "John\'s Age": 30}')

    client = OpenAI_JSON(gpt_api_key="mock-api-key")

    schema = {"name": str, "age": int}
    query = "Generate a JSON object with name and the person's age."

    response = client.handle_request(query, schema)

    # Assert processed data includes the ML-transformed match
    assert response == {
        "name": "John",
        "age": 30,  # Transformed match from ML processor
    }
    assert client.unmatched_data == []  # No unmatched data should remain
    assert client.transformed_data == {
        "John's Age": "age"
    }  # Records ML transformations

    # mock_client.chat.completions.create.assert_called_once_with(
    #     model="gpt-4",
    #     messages=expected_messages(query, expected_system_message),
    #     temperature=0,
    # )
