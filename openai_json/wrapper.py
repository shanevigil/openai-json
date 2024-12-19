from openai_json.schema_handler import SchemaHandler
from openai_json.api_interface import APIInterface
from openai_json.heuristic_processor import HeuristicProcessor
from openai_json.substructure_manager import SubstructureManager
from openai_json.output_assembler import OutputAssembler
from openai_json.ml_processor import MachineLearningProcessor
import json


class Wrapper:
    """
    Coordinates the entire pipeline for processing JSON responses from the OpenAI API.

    The Wrapper integrates multiple components to handle schema validation,
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
        Initializes the Wrapper and its components.

        Args:
            gpt_api_key (str): OpenAI API key for authenticating requests.
            model_path (str, optional): Path to the pre-trained machine learning model. Defaults to None.
            gpt_model (str, optional): OpenAI model to use for queries. Defaults to "gpt-4".
            gpt_temperature (float, optional): Sampling temperature for the model. Defaults to 0.
        """
        self.schema_handler = SchemaHandler()
        self.api_interface = APIInterface(
            gpt_api_key, model=gpt_model, temperature=gpt_temperature
        )
        self.heuristic_processor = HeuristicProcessor(self.schema_handler)
        self.substructure_manager = SubstructureManager()
        self.output_assembler = OutputAssembler()
        self.ml_processor = MachineLearningProcessor(model_path)

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
        # Step 1: Submit schema
        self.schema_handler.submit_schema(schema)

        # Step 2: Send query to OpenAI API
        try:
            raw_response = self.api_interface.send_query(query)
        except RuntimeError as e:
            return {"error": f"Failed to fetch response: {str(e)}"}

        # Step 3: Parse the response (already validated as JSON by send_query)
        try:
            parsed_response = json.loads(raw_response)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON response: {str(e)}"}

        # Step 4: Apply heuristic processing
        processed_data, unmatched_keys = self.heuristic_processor.process(
            parsed_response
        )

        # Step 5: Store unmatched keys in SubstructureManager
        self.substructure_manager.store_unmatched_keys(unmatched_keys, parsed_response)

        # Step 6: Process unmatched data with MachineLearningProcessor
        unmatched_data = self.substructure_manager.retrieve_unmatched_data()
        transformed_data = self.ml_processor.predict_transformations(unmatched_data)

        # Step 7: Assemble the final output
        final_output = self.output_assembler.assemble_output(
            processed_data, transformed_data
        )

        return final_output
