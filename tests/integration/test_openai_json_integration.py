import pytest
from openai_json.openai_json import OpenAI_JSON
import logging


@pytest.mark.parametrize(
    "schema, chatgpt_response, expected_output",
    [
        # Test Case 1: Response Variation 1
        (
            {
                "Key A": {"type": "integer"},
                "Key B": {"type": "string"},
                "Key C": {"type": "list"},
            },
            '{"Key A": 231, "Key B": "Mary has a dog.", "Key C": ["Max"]}',
            {
                "processed_data": {
                    "Key A": 231,
                    "Key B": "Mary has a dog.",
                    "Key C": ["Max"],
                }
            },
        ),
        # Test Case 2: Response Variation 2
        (
            {
                "Key A": {"type": "integer"},
                "Key B": {"type": "string"},
                "Key C": {"type": "list", "items": {"type": "string"}},
            },
            '{"Key A": 231, "Key B": "Mary has a dog.", "Key C": ["Max"]}',
            {
                "processed_data": {
                    "Key A": 231,
                    "Key B": "Mary has a dog.",
                    "Key C": ["Max"],
                }
            },
        ),
        # Test Case 3: Response Variation 3
        (
            {
                "Key A": {"type": "integer"},
                "Key B": {"type": "string"},
                "Key C": {"type": "list", "items": {"type": "string"}},
            },
            '{"Key a": "231", "Key b": "Mary two dogs.", "Key c": "Max, Spot"}',
            {
                "processed_data": {
                    "Key A": 231,
                    "Key B": "Mary two dogs.",
                    "Key C": ["Max", "Spot"],
                }
            },
        ),
        # Test Case 4: Response Variation 4
        (
            {
                "Key A": {"type": "integer"},
                "Key B": {"type": "string"},
                "Key C": {"type": "list", "items": {"type": "string"}},
            },
            '{"Key A": 231.0, "Key B": "Mary some dogs.", "Dogs": {"Key C": ["Max", "Rover", "Spot"]}}',
            {
                "processed_data": {
                    "Key A": 231,
                    "Key B": "Mary some dogs.",
                    "Key C": ["Max", "Rover", "Spot"],
                }
            },
        ),
    ],
)
def test_openai_json_integration(
    mock_openai_client, schema, chatgpt_response, expected_output
):
    """
    Integration test for OpenAI_JSON with various schemas and ChatGPT responses.
    """
    mock_client, set_mock_response, expected_system_message = mock_openai_client

    # Set the mocked response
    set_mock_response(chatgpt_response)

    # Instantiate the OpenAI_JSON wrapper
    openai_json = OpenAI_JSON(gpt_api_key="mock-api-key", model_path=None)

    # Process query and verify output
    query = "Generate a JSON object with Key A, Key B, and Key C."
    output = openai_json.handle_request(query, schema)

    # Debug Logs
    logging.debug("Testing schema: %s", schema)
    logging.debug("Testing response: %s", chatgpt_response)

    # Assert the response matches the expected output
    assert output == expected_output

    # Verify OpenAI client interaction
    mock_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_client.chat.completions.create.call_args
    assert {"role": "system", "content": expected_system_message} in kwargs["messages"]
