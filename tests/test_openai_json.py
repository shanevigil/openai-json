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
    """Test the OpenAI_JSON class with a valid OpenAI API response."""
    mock_client, set_mock_response, expected_system_message = mock_openai_client

    set_mock_response('{"name": "John", "age": 30, "email": "john@example.com"}')

    client = OpenAI_JSON(gpt_api_key="mock-api-key")
    schema = {"name": str, "age": int, "email": str}
    query = "Generate a JSON object with name, age, and email."

    response = client.handle_request(query, schema)

    assert response["processed_data"] == {
        "name": "John",
        "age": 30,
        "email": "john@example.com",
    }
    assert "unmatched_data" not in response

    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=expected_messages(query, expected_system_message),
        temperature=0,
    )


def test_OpenAI_JSON_with_custom_model_and_temperature(
    mock_openai_client, expected_messages
):
    """Test the OpenAI_JSON class with a custom model and temperature."""
    mock_client, set_mock_response, expected_system_message = mock_openai_client
    set_mock_response('{"name": "Alice", "age": 25, "email": "alice@example.com"}')

    client = OpenAI_JSON(
        gpt_api_key="mock-api-key", gpt_model="custom-model", gpt_temperature=0.8
    )

    schema = {"name": str, "age": int, "email": str}
    query = "Generate a JSON object with name, age, and email."

    response = client.handle_request(query, schema)

    assert response["processed_data"] == {
        "name": "Alice",
        "age": 25,
        "email": "alice@example.com",
    }
    assert "unmatched_data" not in response

    mock_client.chat.completions.create.assert_called_once_with(
        model="custom-model",
        messages=expected_messages(query, expected_system_message),
        temperature=0.8,
    )


def test_OpenAI_JSON_with_output_assembler(mock_openai_client, expected_messages):
    mock_client, set_mock_response, expected_system_message = mock_openai_client

    set_mock_response(
        '{"name": "John", "age": 30, "email": "john@example.com", "extra_key": "extra_value"}'
    )

    client = OpenAI_JSON(gpt_api_key="mock-api-key")

    schema = {"name": str, "age": int, "email": str}
    query = "Generate a JSON object with name, age, email, and extra data."

    response = client.handle_request(query, schema)

    assert response["processed_data"] == {
        "name": "John",
        "age": 30,
        "email": "john@example.com",
    }
    assert response["unmatched_data"] == {"extra_key": "extra_value"}

    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=expected_messages(query, expected_system_message),
        temperature=0,
    )


@patch("openai_json.ml_processor.MachineLearningProcessor.load_model")
@patch("openai_json.ml_processor.MachineLearningProcessor.predict_transformations")
def test_OpenAI_JSON_with_ml_processor(
    mock_predict, mock_load_model, mock_openai_client, expected_messages
):
    mock_client, set_mock_response, expected_system_message = mock_openai_client

    set_mock_response('{"name": "John", "age": 30, "extra_key": "extra_value"}')

    mock_predict.return_value = {"extra_key": "transformed_value"}

    client = OpenAI_JSON(gpt_api_key="mock-api-key", model_path="mock-model.pkl")

    schema = {"name": str, "age": int}
    query = "Generate a JSON object with name, age, and extra data."

    response = client.handle_request(query, schema)

    assert response["processed_data"] == {"name": "John", "age": 30}
    assert response["unmatched_data"] == {"extra_key": "transformed_value"}

    mock_client.chat.completions.create.assert_called_once_with(
        model="gpt-4",
        messages=expected_messages(query, expected_system_message),
        temperature=0,
    )
    mock_predict.assert_called_once_with({"extra_key": "extra_value"})
