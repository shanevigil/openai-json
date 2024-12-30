import logging
import json
from openai_json.schema_handler import SchemaHandler
from openai_json.data_manager import DataManager, ResultData
from openai_json.api_interface import APIInterface, AsyncAPIInterface
from openai_json.heuristic_processor import HeuristicProcessor
from openai_json.ml_processor import MachineLearningProcessor
import asyncio


class OpenAI_JSON:
    """
    Coordinates the entire pipeline for processing JSON responses from the OpenAI API.

    The OpenAI_JSON is wrapper for the python openai module designed to streamline
    ChatGPT outputs that are expected to be in JSON format. Due to the nature of the GPT
    model, outputs are expected to vary slightly which can be problematic for developers
    OpenAI_JSON integrates multiple components to handle schema validation,
    API interactions, heuristic and machine learning-based processing, and
    final output assembly. It serves as the primary interface for end-to-end
    structured JSON handling.

    Attributes:
        schema_handler (SchemaHandler): Manages schema validation and normalization.
        api_interface (APIInterface): For synchronous API interactions.
        async_api_interface (AsyncAPIInterface): For asynchronous API interactions.
        heuristic_processor (HeuristicProcessor): Applies heuristic rules to process JSON.
        ml_processor (MachineLearningProcessor): Applies ML transformations to unmatched data.
        data_manager (DataManager): Manages and consolidates processing results.
    """

    def __init__(
        self,
        gpt_api_key: str,
        schema: str or dict = None,
        gpt_model: str = "gpt-4",
        gpt_temperature: float = 0,
    ):
        """
        Initializes the OpenAI_JSON class and its components.

        This method sets up the key components of the JSON processing pipeline,
        including schema handling, OpenAI API interactions, heuristic processing,
        machine learning-based transformations, and output assembly.

        Args:
            gpt_api_key (str): The API key to authenticate with the OpenAI API.
            schema (str or dict, optional): A JSON schema, either as a dictionary
                or a JSON-formatted string, to validate and process data against.
                If provided, it will be submitted to the `SchemaHandler.`
            model_path (str, optional): The path to the pre-trained machine learning model.
                If provided, the model is loaded for schema-compliant transformations.
            gpt_model (str, optional): The name of the OpenAI GPT model to use.
                Defaults to "gpt-4".
            gpt_temperature (float, optional): The temperature for controlling
                the randomness of the GPT model's responses. Defaults to 0.

        Attributes:
            schema_handler (SchemaHandler): Manages schema submission, normalization,
                and validation. If a `schema` is provided during initialization,
                it is submitted to this handler. Accepts either a dictionary or
                a JSON-formatted string.
            unmatched_data (list): A list of keys and their associated data that did not
                conform to the schema during processing. Useful for debugging or handling
                non-compliant data in the response from ChatGPT.
            errors (list): A list of keys and their associated data that were of an
                unexpected type and failed to be coerced into the correct type in the
                response from ChatGPT.
            validation_error (str): A message describing any validation error encountered
                during schema submission or response handling. If no validation errors
                occurred, this will be `None`.
            api_interface (APIInterface): Handles interactions with the OpenAI API,
                including sending queries and receiving responses.
            heuristic_processor (HeuristicProcessor): Applies heuristic rules to
                process and align JSON data with the schema.
            substructure_manager (SubstructureManager): Manages and stores unmatched
                keys and their associated data for further analysis.
            output_assembler (OutputAssembler): Combines processed data, unmatched keys,
                and machine learning transformations into the final output.
            ml_processor (MachineLearningProcessor): Applies machine learning predictions
                to align unmatched data with the schema. If `model_path` is not provided,
                this component is initialized without a model.

        """

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing OpenAI_JSON with OpenAI API and ML components.")

        self.gpt_api_key = gpt_api_key
        self.gpt_model = gpt_model
        self.gpt_temperature = gpt_temperature

        self.schema_handler = SchemaHandler(schema)
        if schema:
            self._init_apis()

        self.heuristic_processor = HeuristicProcessor(self.schema_handler)
        self.ml_processor = MachineLearningProcessor(self.schema_handler)
        self.data_manager = DataManager(self.schema_handler)

        self.unmatched_data = {}
        self.errors = {}
        self.validation_error = None

        self.logger.info("OpenAI_JSON initialization complete.")

    def _init_apis(self):
        self.example_json_string = self.schema_handler.generate_example_json()
        system_message = f"Respond in valid JSON format. Use the following example JSON as a reference:\n{self.example_json_string}"

        self.api_interface = APIInterface(
            self.gpt_api_key,
            model=self.gpt_model,
            temperature=self.gpt_temperature,
            system_message=system_message,
        )
        self.async_api_interface = AsyncAPIInterface(
            self.gpt_api_key,
            model=self.gpt_model,
            temperature=self.gpt_temperature,
            system_message=system_message,
        )

    def request(self, query: str, schema: dict = None) -> dict:
        """
        Sends a query to the OpenAI API and processes the response synchronously.

        Args:
            query (str): The query to send.
            schema (dict, optional): Schema for response validation.

        Returns:
            dict: Schema-compliant response or error details.

        Workflow:
            1. Submit the schema for validation.
            2. Extract prompts and append them to the query.
            3. Send the query to the OpenAI API and retrieve the raw response.
            4. Parse the raw response into JSON format.
            5. Apply heuristic rules to align data with the schema, using data-type coercion where possible.
            6. Identify and store unmatched keys using SubstructureManager.
            7. Use MachineLearningProcessor to predict transformations for unmatched data.
            8. Combine processed and transformed data into the final output.
        """

        full_query = self._prepare_query(query, schema)

        try:
            raw_response = self.api_interface.send_query(full_query)
            return self._process_response(raw_response)

        except Exception as e:
            self.logger.error("Synchronous request failed: %s", e)
            return {}

    async def async_request(self, query: str, schema: dict = None) -> dict:
        """
        Sends a query to the OpenAI API and processes the response asynchronously.

        Args:
            query (str): The query to send.
            schema (dict, optional): Schema for response validation.

        Returns:
            dict: Schema-compliant response or error details.

        Workflow:
            1. Submit the schema for validation.
            2. Extract prompts and append them to the query.
            3. Send the query to the OpenAI API and retrieve the raw response.
            4. Parse the raw response into JSON format.
            5. Apply heuristic rules to align data with the schema, using data-type coercion where possible.
            6. Identify and store unmatched keys using SubstructureManager.
            7. Use MachineLearningProcessor to predict transformations for unmatched data.
            8. Combine processed and transformed data into the final output.
        """
        full_query = self._prepare_query(query, schema)

        try:
            raw_response = await self.async_api_interface.send_query(full_query)
            return self._process_response(raw_response)

        except Exception as e:
            self.logger.error("Asynchronous request failed: %s", e)
            return {}

    def _prepare_query(self, query: str, schema: dict = None) -> str:
        """
        Prepare the full query with prompts and example JSON.
        """
        try:
            if schema:
                self.schema_handler.submit_schema(schema)
                self._init_apis()
            prompts_string = self.schema_handler.extract_prompts()
            query_with_prompts = f"{query}\n\n{prompts_string}"
            instructions_string = (
                "\n\nPlease ensure the response adheres to the following schema:\n"
            )
            return (
                f"{query_with_prompts}{instructions_string}{self.example_json_string}"
            )
        except Exception as e:
            raise ValueError(f"Failed to prepare query: {e}")

    def _process_response(self, response: str) -> dict:
        """
        Process the raw response from OpenAI.
        """
        try:
            parsed_response = json.loads(response)
            self.data_manager.add_result(ResultData(unmatched=parsed_response))
            self.data_manager.add_result(
                self.heuristic_processor.process(self.data_manager.unmatched)
            )
            self.data_manager.add_result(
                self.ml_processor.process(self.data_manager.unmatched)
            )
            final_output = self.data_manager.finalize_output()

            self.unmatched_data = self.data_manager.unmatched
            self.errors = self.data_manager.errors
            return final_output
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            raise RuntimeError(f"Processing failed: {e}")
