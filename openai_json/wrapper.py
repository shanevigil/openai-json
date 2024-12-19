from openai_json.schema_handler import SchemaHandler
from openai_json.api_interface import APIInterface
from openai_json.heuristic_processor import HeuristicProcessor


class Wrapper:
    """Integrates all components for end-to-end processing."""

    def __init__(self, api_key):
        self.schema_handler = SchemaHandler()
        self.api_interface = APIInterface(api_key)
        self.heuristic_processor = HeuristicProcessor(self.schema_handler)

    def handle_request(self, query: str, schema: dict):
        """Handles the entire pipeline: query -> response -> validation."""
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

        return {"processed_data": processed_data, "unmatched_keys": unmatched_keys}
