from unittest.mock import patch, MagicMock
import pytest
from openai_json.wrapper import Wrapper


@patch("openai.OpenAI")
def test_wrapper_valid(mock_create):
    mock_create.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"name": "John", "age": 30, "email": "john@example.com"}'
                }
            }
        ]
    }

    api_key = "mock-api-key"
    wrapper = Wrapper(api_key)
    schema = {"name": str, "age": int, "email": str}
    query = "Generate a JSON object with name, age, and email."

    response = wrapper.handle_request(query, schema)

    # Ensure processed_data contains the correct data
    assert response["processed_data"] == {
        "name": "John",
        "age": 30,
        "email": "john@example.com",
    }

    # Ensure unmatched_data is empty as there are no extra keys
    assert "unmatched_data" not in response


@patch("openai.OpenAI")
def test_wrapper_with_output_assembler(mock_create):
    mock_create.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"name": "John", "age": 30, "email": "john@example.com", "extra_key": "extra_value"}'
                }
            }
        ]
    }

    api_key = "mock-api-key"
    wrapper = Wrapper(api_key)
    schema = {"name": str, "age": int, "email": str}
    query = "Generate a JSON object with name, age, email, and extra data."

    response = wrapper.handle_request(query, schema)
    assert response["processed_data"] == {
        "name": "John",
        "age": 30,
        "email": "john@example.com",
    }
    assert response["unmatched_data"] == {"extra_key": "extra_value"}


@patch("openai_json.ml_processor.MachineLearningProcessor.load_model")
@patch("openai_json.ml_processor.MachineLearningProcessor.predict_transformations")
@patch("openai.OpenAI")
def test_wrapper_with_ml_processor(mock_openai, mock_predict, mock_load_model):
    """Test the Wrapper class with mocked ML processor and OpenAI API."""
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Mock API response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"name": "John", "age": 30, "extra_key": "extra_value"}'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Mock predictions for unmatched keys
    mock_predict.return_value = {"extra_key": "transformed_value"}

    wrapper = Wrapper(api_key="mock-api-key", model_path="mock-model.pkl")

    schema = {"name": str, "age": int}
    query = "Generate a JSON object with name, age, and extra data."

    response = wrapper.handle_request(query, schema)

    # Assertions
    assert response["processed_data"] == {"name": "John", "age": 30}
    assert response["unmatched_data"] == {"extra_key": "transformed_value"}

    mock_client.chat.completions.create.assert_called_once()
    mock_predict.assert_called_once_with({"extra_key": "extra_value"})
