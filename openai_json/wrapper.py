from openai_json.schema_handler import SchemaHandler
from openai_json.api_interface import APIInterface
from openai_json.heuristic_processor import HeuristicProcessor
from openai_json.substructure_manager import SubstructureManager
from openai_json.output_assembler import OutputAssembler
from openai_json.ml_processor import MachineLearningProcessor


class Wrapper:
    """
    Integrates all components for end-to-end processing.
    """

    def __init__(self, api_key, model_path=None):
        self.schema_handler = SchemaHandler()
        self.api_interface = APIInterface(api_key)
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
        raw_response = self.api_interface.send_query(query)

        # Step 3: Parse the response
        try:
            parsed_response = self.api_interface.parse_response(raw_response)
        except ValueError as e:
            return {"error": str(e)}

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
