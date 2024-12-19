import logging
from openai_json.schema_handler import SchemaHandler
from openai_json.api_interface import APIInterface
from openai_json.heuristic_processor import HeuristicProcessor
from openai_json.substructure_manager import SubstructureManager
from openai_json.output_assembler import OutputAssembler
from openai_json.ml_processor import MachineLearningProcessor
import json


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
        schema_handler (SchemaHandler): Manages schema submission and validation.
        api_interface (APIInterface): Handles interactions with the OpenAI API.
        heuristic_processor (HeuristicProcessor): Processes data using heuristic rules.
        substructure_manager (SubstructureManager): Stores unmatched keys and their data.
        output_assembler (OutputAssembler): Combines processed and transformed data.
        ml_processor (MachineLearningProcessor): Predicts schema-compliant transformations.
    """

    def __init__(
        self,
        gpt_api_key: str,
        model_path: str = None,
        gpt_model: str = "gpt-4",
        gpt_temperature: float = 0,
    ):
        """
        Initializes the OpenAI_JSON and its components.

        Args:
            gpt_api_key (str): OpenAI API key for authenticating requests.
            model_path (str, optional): Path to the pre-trained machine learning model. Defaults to None.
            gpt_model (str, optional): OpenAI model to use for queries. Defaults to "gpt-4".
            gpt_temperature (float, optional): Sampling temperature for the model. Defaults to 0.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing OpenAI_JSON with OpenAI API and ML components.")

        self.schema_handler = SchemaHandler()
        self.api_interface = APIInterface(
            gpt_api_key, model=gpt_model, temperature=gpt_temperature
        )
        self.heuristic_processor = HeuristicProcessor(self.schema_handler)
        self.substructure_manager = SubstructureManager()
        self.output_assembler = OutputAssembler()
        self.ml_processor = MachineLearningProcessor(model_path)

        self.logger.info("OpenAI_JSON initialization complete.")

    def handle_request(self, query: str, schema: dict) -> dict:
        """
        Processes a query end-to-end: sends the query to the OpenAI API, validates and processes
        the response, and assembles a schema-compliant output.

        Args:
            query (str): The query string to send to the OpenAI API.
            schema (dict): The schema dict defining the expected JSON structure.

        Returns:
            dict: The final schema-compliant JSON response, or an error dictionary if processing fails.

        Workflow:
            1. Submit the schema for validation.
            2. Send the query to the OpenAI API and retrieve the raw response.
            3. Parse the raw response into JSON format.
            4. Apply heuristic rules to align data with the schema.
            5. Identify and store unmatched keys using SubstructureManager.
            6. Use MachineLearningProcessor to predict transformations for unmatched data.
            7. Combine processed and transformed data into the final output.
        """
        self.logger.info("Starting request handling.")
        self.logger.debug("Query: %s", query)
        self.logger.debug("Schema: %s", schema)

        # Step 1: Submit schema
        try:
            self.schema_handler.submit_schema(schema)
            self.logger.info("Schema submitted successfully.")
        except ValueError as e:
            self.logger.error("Schema submission failed: %s", e)
            return {"error": f"Schema error: {str(e)}"}

        # Step 2: Send query to OpenAI API
        try:
            raw_response = self.api_interface.send_query(query)
            self.logger.info("Received raw response from OpenAI API.")
        except RuntimeError as e:
            self.logger.error("Failed to fetch response from OpenAI API: %s", e)
            return {"error": f"Failed to fetch response: {str(e)}"}

        # Step 3: Parse the response
        try:
            parsed_response = json.loads(raw_response)
            self.logger.debug("Parsed JSON response: %s", parsed_response)
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON response: %s", e)
            return {"error": f"Failed to parse JSON response: {str(e)}"}

        # Step 4: Apply heuristic processing
        try:
            processed_data, unmatched_keys = self.heuristic_processor.process(
                parsed_response
            )
            self.logger.info("Heuristic processing completed.")
            self.logger.debug("Processed data: %s", processed_data)
            self.logger.debug("Unmatched keys: %s", unmatched_keys)
        except Exception as e:
            self.logger.error("Heuristic processing failed: %s", e)
            return {"error": f"Heuristic processing failed: {str(e)}"}

        # Step 5: Store unmatched keys
        self.substructure_manager.store_unmatched_keys(unmatched_keys, parsed_response)
        self.logger.info("Unmatched keys stored.")

        # Step 6: Process unmatched data with ML
        unmatched_data = self.substructure_manager.retrieve_unmatched_data()
        transformed_data = self.ml_processor.predict_transformations(unmatched_data)
        self.logger.info("ML processing completed.")
        self.logger.debug("Transformed data: %s", transformed_data)

        # Step 7: Assemble the final output
        final_output = self.output_assembler.assemble_output(
            processed_data, transformed_data
        )
        self.logger.info("Final output assembled.")
        self.logger.debug("Final output: %s", final_output)

        return final_output
