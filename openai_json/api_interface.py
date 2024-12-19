import json
import openai


class APIInterface:
    """Handles interactions with OpenAI's ChatGPT API."""

    def __init__(self, api_key: str):
        openai.api_key = api_key

    def send_query(self, query: str):
        """Sends a query to the ChatGPT API."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4", messages=[{"role": "user", "content": query}]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return {"error": str(e)}

    def parse_response(self, response: str):
        """Parses the API response into a Python dictionary."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
