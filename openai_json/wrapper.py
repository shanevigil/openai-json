from openai_json.schema_handler import SchemaHandler
from openai_json.api_interface import APIInterface
from openai_json.heuristic_processor import HeuristicProcessor
from openai_json.substructure_manager import SubstructureManager
from openai_json.output_assembler import OutputAssembler
from openai_json.ml_processor import MachineLearningProcessor
import json


class Wrapper:
    """
    Integrates all components for end-to-end processing.
    """

    def __init__(
        self, gpt_api_key, model_path=None, gpt_model="gpt-4", gpt_temperature=0
    ):
        self.schema_handler = SchemaHandler()
        self.api_interface = APIInterface(
            gpt_api_key, model=gpt_model, temperature=gpt_temperature
        )
        self.heuristic_processor = HeuristicProcessor(self.schema_handler)
        self.substructure_manager = SubstructureManager()
        self.output_assembler = OutputAssembler()
        self.ml_processor = MachineLearningProcessor(model_path)

    def handle_request(self, query: str, schema: dict):
        """
        Handles the entire pipeline: query -> response -> validation.
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
