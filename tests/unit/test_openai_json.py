import pytest
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
    sync_mock_client, _, set_mock_response, _ = mock_openai_client

    # Prepare the schema and query
    schema = {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
        "addres": {"type": "string"},
    }
    query = "Generate a JSON object with name, age, and email."

    # Mock response from OpenAI API
    set_mock_response(
        '{"name": "John", "age": 30, "email": "john@example.com","address":"4 privet drive"}'
    )

    # Create OpenAI_JSON instance
    client = OpenAI_JSON(gpt_api_key="mock-api-key")
    client.api_client = sync_mock_client

    # Run the handle_request method
    response = client.handle_request(query, schema)

    # Assertions for the response
    assert response == {
        "name": "John",
        "age": 30,
        "email": "john@example.com",
        "addres": "4 privet drive",  # Test fuzzy matching
    }
    assert client.unmatched_data == {}
    assert client.errors == {}


def test_OpenAI_JSON_with_unmatched_data(mock_openai_client, expected_messages):
    """Test OpenAI_JSON handling of unmatched data."""
    sync_mock_client, _, set_mock_response, _ = mock_openai_client

    schema = {"First Name": {"type": "string"}, "Age": {"type": "integer"}}
    query = "Generate a JSON object with name and age."

    # Mock response with extra fields
    set_mock_response('{"first_name": "Alice", "age": 25, "extra": "unexpected"}')

    client = OpenAI_JSON(gpt_api_key="mock-api-key")
    client.api_client = sync_mock_client

    response = client.handle_request(query, schema)

    assert response == {"First Name": "Alice", "Age": 25}
    assert client.unmatched_data == {"extra": "unexpected"}
    assert client.errors == {}


def test_OpenAI_JSON_with_errors(mock_openai_client, expected_messages):
    """Test OpenAI_JSON with inputs that produce errors during heuristic processing."""
    sync_mock_client, _, set_mock_response, _ = mock_openai_client

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

    client = OpenAI_JSON(gpt_api_key="mock-api-key")
    client.api_client = sync_mock_client

    # Run the handle_request method
    response = client.handle_request(query, schema)

    # Assertions for the response
    assert response == {
        "name": "Alice",
        "email": "alice@someplace.com",
        "age": 25,
    }  # Only valid data should be returned
    assert client.unmatched_data == {}
    assert client.errors == {}


def test_OpenAI_JSON_with_system_message(mock_openai_client, schema_handler):
    """Test OpenAI_JSON to ensure the system message includes the example JSON and schema prompts."""
    sync_mock_client, _, set_mock_response, expected_system_message_base = (
        mock_openai_client
    )

    # Prepare schema and query
    schema = {
        "name": {"type": "string", "prompt": "The full given name"},
        "age": {"type": "integer", "prompt": "The age of the famous person"},
        "email": {"type": "string", "prompt": "The personal email address"},
    }
    query = "Who was the most famous person in 1950?"

    # Use the SchemaHandler to generate example JSON
    schema_handler.submit_schema(schema)
    example_json = schema_handler.generate_example_json()

    # Construct the expected system message
    expected_system_message = f"{expected_system_message_base} Use the following example JSON as a reference:\n{example_json}"

    # Construct the combined query with schema prompts
    schema_prompts = (
        "Here are the field-specific instructions:\n"
        "name: The full given name\n"
        "age: The age of the famous person\n"
        "email: The personal email address"
    )
    combined_query = f"{query}\n\n{schema_prompts}\n\nPlease ensure the response adheres to the following schema:\n{example_json}"

    # Mock response content
    mock_content = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
    set_mock_response(mock_content)  # Configure the mock response

    client = OpenAI_JSON(gpt_api_key="mock-api-key", schema=schema)
    client.api_client = sync_mock_client

    # Run the handle_request method
    response = client.handle_request(query, schema)

    # Verify the API was called with the expected payload
    sync_mock_client.chat.completions.create.assert_called_once()
    called_args = sync_mock_client.chat.completions.create.call_args[
        1
    ]  # Extract call arguments

    # Validate the system message
    assert called_args["messages"][0] == {
        "role": "system",
        "content": expected_system_message,
    }, "System message not included in API payload."

    # Validate the user query with schema prompts and example JSON
    assert called_args["messages"][1] == {
        "role": "user",
        "content": combined_query,
    }, "User query with schema prompts and example JSON not included in API payload."

    # Verify the final response matches the mocked content
    assert response == {
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com",
    }
