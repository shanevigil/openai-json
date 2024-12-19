import requests
import json
import logging
import openai


class APIInterface:
    """Handles interactions with OpenAI's ChatGPT API."""

    def __init__(
        self,
        api_key: str,
        model="gpt-4",
        retries=3,
        system_message="Respond in valid JSON format.",
        temperature=0,
        schema=None,
    ):
        """
        Initializes the API interface.

        Args:
            api_key (str): API key for OpenAI.
            model (str): OpenAI model to use (default: "gpt-4").
            retries (int): Number of retries for API calls (default: 3).
            system_message (str): System message to guide ChatGPT responses.
            temperature (float): Sampling temperature for the model (default: 0).
            schema (dict): JSON schema for validation of responses.
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.retries = retries
        self.system_message = system_message
        self.temperature = temperature
        self.schema = schema

        self.logger = logging.getLogger(__name__)  # Use existing logger configuration

    def send_query(self, query: str):
        """
        Sends a query to the ChatGPT API.

        Args:
            query (str): User query to send to the API.

        Returns:
            str: The response content from the API.

        Raises:
            RuntimeError: If the API call fails after retries.
        """
        self.logger.debug("Sending query: %s", query)

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": query},
        ]
        retries = self.retries

        while retries > 0:
            try:
                self.logger.debug("Attempting API call. Retries remaining: %d", retries)

                # Send the request
                response = openai.OpenAI(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )

                # Log API response metadata
                self.logger.info(
                    "API call successful. Usage: %s", response.get("usage", {})
                )

                # Extract response content
                content = response.choices[0].message.content.strip()
                self.logger.debug("Raw response content: %s", content)

                # Verify JSON validity
                try:
                    json.loads(content)
                    return content
                except json.JSONDecodeError:
                    self.logger.warning("Received invalid JSON response: %s", content)
                    raise ValueError("Invalid JSON response")

            except (requests.exceptions.RequestException, ValueError) as e:
                self.logger.error("Error occurred: %s", e)
                retries -= 1
                if retries == 0:
                    self.logger.critical(
                        "API query failed after %d retries: %s", self.retries, e
                    )
                    raise RuntimeError(
                        f"API query failed after {self.retries} retries: {e}"
                    )

            except Exception as e:
                if self._is_retryable_error(e):
                    self.logger.warning("Retryable error occurred: %s", e)
                else:
                    self.logger.error("Non-retryable error occurred: %s", e)
                    raise

                retries -= 1
                if retries == 0:
                    self.logger.critical(
                        "API query failed after %d retries: %s", self.retries, e
                    )
                    raise RuntimeError(
                        f"API query failed after {self.retries} retries: {e}"
                    )

    def _is_retryable_error(self, error):
        """
        Determines if an error is retryable.

        Args:
            error (Exception): The exception to evaluate.

        Returns:
            bool: True if the error is retryable, False otherwise.
        """
        retryable_errors = [
            "429 Too Many Requests",
            "500 Internal Server Error",
            "503 Service Unavailable",
            "504 Gateway Timeout",
        ]
        return any(code in str(error) for code in retryable_errors)
