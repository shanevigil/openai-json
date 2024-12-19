from unittest.mock import patch
from openai_json.wrapper import Wrapper


@patch("openai.ChatCompletion.create")
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
    assert response["processed_data"] == {
        "name": "John",
        "age": 30,
        "email": "john@example.com",
    }
    assert response["unmatched_keys"] == []
