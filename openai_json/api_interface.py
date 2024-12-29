import asyncio
import json
import logging
from openai import OpenAI, AsyncOpenAI, RateLimitError


class BaseAPIInterface:
    """
    Base class for shared functionality between synchronous and asynchronous API interfaces.
    """

    def __init__(
        self,
        api_key,
        model="gpt-4",
        system_message="Respond in valid JSON format.",
        temperature=0,
    ):
        self.api_key = api_key
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    def _prepare_payload(self, query):
        """
        Prepares the payload for the API request.
        """
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": query},
            ],
            "temperature": self.temperature,
        }

    def _validate_json(self, content):
        """
        Validates the response content as JSON.
        """
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON received: %s", content)
            raise ValueError("Invalid JSON response") from e

    def _is_retryable_error(self, error):
        retryable_errors = [
            "429 Too Many Requests",
            "500 Internal Server Error",
            "503 Service Unavailable",
            "504 Gateway Timeout",
        ]
        # Check for specific error messages or retryable OpenAI exceptions
        if isinstance(error, RateLimitError):
            return True

        return any(code in str(error) for code in retryable_errors)

    def _retry_request_sync(self, request_func, retries):
        """
        Handles retry logic for synchronous API requests by wrapping the unified retry mechanism.

        Args:
            request_func (callable): A callable that performs the API request and returns a response.
            retries (int): Number of retries allowed for transient errors.

        Returns:
            Any: The result of the successful request_func call.

        Raises:
            Exception: If retries are exhausted or a non-retryable error occurs.
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new loop if the current one is running (e.g., during tests)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._retry_request(request_func, retries, is_async=False)
        )

    async def _retry_request(self, request_func, retries, is_async=True):
        """
        Handles retry logic for both synchronous and asynchronous API requests.

        Args:
            request_func (callable): A callable or awaitable that performs the API request and returns a response.
            retries (int): Number of retries allowed for transient errors.
            is_async (bool): If True, treats request_func as an awaitable for asynchronous operations.

        Returns:
            Any: The result of the successful request_func call.

        Raises:
            Exception: If retries are exhausted or a non-retryable error occurs.
        """
        while retries > 0:
            try:
                if is_async:
                    return await request_func()
                else:
                    return request_func()
            except Exception as e:
                self.logger.error(
                    "Error during %s API call: %s", "async" if is_async else "sync", e
                )

                # Treat ValueError from _validate_json as retryable
                if isinstance(e, ValueError):
                    self.logger.warning(
                        "Invalid JSON response received. Retrying... (%d retries left)",
                        retries - 1,
                    )
                elif not self._is_retryable_error(e):
                    self.logger.error("Non-retryable error encountered: %s", e)
                    raise  # Non-retryable error; fail immediately

                retries -= 1
                self.logger.warning(
                    "Retrying %s request... %d retries remaining",
                    "async" if is_async else "sync",
                    retries,
                )
                if retries == 0:
                    raise RuntimeError(
                        f"{'Async' if is_async else 'Sync'} API query failed after retries: {e}"
                    ) from e


class APIInterface(BaseAPIInterface):
    """
    A class to handle interactions with OpenAI's ChatGPT API, focusing on generating JSON outputs.

    This class provides methods to communicate with OpenAI's API, including retry logic
    and validation of JSON responses. It is designed to guide the API towards structured outputs
    using system-level instructions and allows fine-tuning of API behavior via parameters.
    """

    def __init__(self, api_key, model="gpt-4", retries=3, **kwargs):
        """
        Initialize the API interface.

        Args:
            api_key (str): API key for OpenAI to authenticate requests.
            model (str, optional): The OpenAI model to use for generating responses.
                Defaults to "gpt-4".
            retries (int, optional): The number of retry attempts for API calls in case of failure.
                Defaults to 3.
            system_message (str, optional): A system-level message to guide the ChatGPT model
                towards specific behavior or output formats. Defaults to "Respond in valid JSON format.".
            temperature (float, optional): Sampling temperature for the model, controlling randomness
                in responses. A value closer to 0 produces deterministic outputs, while higher values
                generate more diverse outputs. Defaults to 0.0.

        Attributes:
            client (openai.OpenAI): OpenAI client initialized with the provided API key.
            model (str): The OpenAI model to be used for API calls.
            retries (int): Number of retry attempts for API calls.
            system_message (str): Instruction message to guide the model's output.
            temperature (float): Sampling temperature for response generation.
            logger (logging.Logger): Logger instance for logging API interactions and errors.
        """
        super().__init__(api_key, model, **kwargs)
        self.client = OpenAI(api_key=api_key)
        self.retries = retries

        # TODO Keep the following until tests run correctly, then delete as it is in the base
        # self.client = openai.OpenAI(api_key=api_key)
        # self.model = model
        # self.retries = retries
        # self.system_message = system_message
        # self.temperature = temperature
        # self.logger = logging.getLogger(__name__)  # Use existing logger configuration

    def send_query(self, query: str):
        """
        Send a query to the ChatGPT API and handle the response.

        This method communicates with the OpenAI ChatGPT API using the provided query, handles
        retry logic for transient errors, and validates the API response for JSON compliance.

        Args:
            query (str): The user-provided query or prompt to send to the ChatGPT API.

        Returns:
            str: The raw content of the response from the ChatGPT API, validated as a JSON-compatible string.

        Raises:
            ValueError: If the API returns a response that is not valid JSON and retries are exhausted.
            RuntimeError: If the API call fails after exhausting all retry attempts due to transient errors.
        """
        payload = self._prepare_payload(query)

        def request_func():
            response = self.client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            self._validate_json(content)
            return content

        return self._retry_request_sync(request_func, self.retries)

        # TODO Delete the following after verifying tests work
        # self.logger.debug("Sending query: %s", query)

        # messages = [
        #     {"role": "system", "content": self.system_message},
        #     {"role": "user", "content": query},
        # ]
        # retries = self.retries

        # while retries > 0:
        #     try:
        #         self.logger.debug("Attempting API call. Retries remaining: %d", retries)

        #         # Send the request
        #         response = self.client.chat.completions.create(
        #             model=self.model,
        #             messages=messages,
        #             temperature=self.temperature,
        #         )

        #         # Log API response metadata
        #         self.logger.info(
        #             "API call successful. Usage: %s", response.get("usage", {})
        #         )

        #         # Extract response content
        #         content = response.choices[0].message.content.strip()
        #         self.logger.debug("Raw response content: %s", content)

        #         # Verify JSON validity
        #         json.loads(content)
        #         return content

        #     except json.JSONDecodeError as e:
        #         self.logger.warning(
        #             "Invalid JSON response received. Retrying... (%d retries left)",
        #             retries - 1,
        #         )
        #         retries -= 1
        #         if retries == 0:
        #             self.logger.error(
        #                 "Exhausted retries due to invalid JSON: %s", str(e)
        #             )
        #             raise ValueError("Invalid JSON response after retries") from e

        #     except requests.exceptions.RequestException as e:
        #         self.logger.error("HTTP error occurred: %s", e)
        #         retries -= 1
        #         if retries == 0:
        #             self.logger.critical(
        #                 "API query failed after %d retries: %s", self.retries, e
        #             )
        #             raise RuntimeError(
        #                 f"API query failed after {self.retries} retries: {e}"
        #             )

        #     except Exception as e:
        #         retries -= 1

        #         if self._is_retryable_error(e):
        #             self.logger.warning("Retryable error occurred: %s", e)
        #         else:
        #             self.logger.error("Non-retryable error occurred: %s", e)
        #             if retries == 0:
        #                 self.logger.critical(
        #                     "API query failed after %d retries: %s", self.retries, e
        #                 )
        #                 raise RuntimeError(
        #                     f"API query failed after {self.retries} retries: {e}"
        #                 )
        #             continue  # Proceed to the next retry

        #         if retries == 0:
        #             self.logger.critical(
        #                 "API query failed after %d retries: %s", self.retries, e
        #             )
        #             raise RuntimeError(
        #                 f"API query failed after {self.retries} retries: {e}"
        #             )

    # def _is_retryable_error(self, error):
    #     """
    #     Determines if an error is retryable.

    #     Args:
    #         error (Exception): The exception to evaluate.

    #     Returns:
    #         bool: True if the error is retryable, False otherwise.
    #     """
    #     retryable_errors = [
    #         "429 Too Many Requests",
    #         "500 Internal Server Error",
    #         "503 Service Unavailable",
    #         "504 Gateway Timeout",
    #     ]
    #     return any(code in str(error) for code in retryable_errors)


class AsyncAPIInterface(BaseAPIInterface):
    """
    A class to handle asynchronous interactions with OpenAI's ChatGPT API, focusing on generating JSON outputs.

    This class provides methods to communicate with OpenAI's API in an asynchronous manner, including retry logic
    and validation of JSON responses. It is designed to guide the API towards structured outputs using system-level
    instructions and allows fine-tuning of API behavior via parameters.
    """

    def __init__(self, api_key, retries=3, **kwargs):
        """
        Initialize the asynchronous API interface.

        Args:
            api_key (str): API key for OpenAI to authenticate requests.
            model (str, optional): The OpenAI model to use for generating responses.
                Defaults to "gpt-4".
            retries (int, optional): The number of retry attempts for API calls in case of failure.
                Defaults to 3.
            system_message (str, optional): A system-level message to guide the ChatGPT model
                towards specific behavior or output formats. Defaults to "Respond in valid JSON format.".
            temperature (float, optional): Sampling temperature for the model, controlling randomness
                in responses. A value closer to 0 produces deterministic outputs, while higher values
                generate more diverse outputs. Defaults to 0.0.

        Attributes:
            client (openai.AsyncOpenAI): Asynchronous OpenAI client initialized with the provided API key.
            model (str): The OpenAI model to be used for API calls.
            retries (int): Number of retry attempts for API calls.
            system_message (str): Instruction message to guide the model's output.
            temperature (float): Sampling temperature for response generation.
            logger (logging.Logger): Logger instance for logging API interactions and errors.
        """
        super().__init__(api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
        self.retries = retries

    async def send_query(self, query):
        """
        Send a query to the ChatGPT API asynchronously and handle the response.

        This method communicates with the OpenAI ChatGPT API asynchronously using the provided query,
        handles retry logic for transient errors, and validates the API response for JSON compliance.

        Args:
            query (str): The user-provided query or prompt to send to the ChatGPT API.

        Returns:
            str: The raw content of the response from the ChatGPT API, validated as a JSON-compatible string.

        Raises:
            ValueError: If the API returns a response that is not valid JSON and retries are exhausted.
            RuntimeError: If the API call fails after exhausting all retry attempts due to transient errors.
        """
        payload = self._prepare_payload(query)

        async def request_func():
            response = await self.client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            self._validate_json(content)
            return content

        return await self._retry_request(request_func, self.retries, is_async=True)
