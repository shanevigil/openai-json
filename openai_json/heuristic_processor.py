import logging
from openai_json.schema_handler import SchemaHandler


class HeuristicProcessor:
    """
    Applies heuristic rules to process and match JSON structures to schemas.

    This class is responsible for aligning straightforward elements of JSON
    responses to their corresponding schema using predefined rules.

    Attributes:
        schema_handler (SchemaHandler): An instance of SchemaHandler for schema validation.
    """

    def __init__(self, schema_handler: SchemaHandler):
        """
        Initializes the HeuristicProcessor with a schema handler.

        Args:
            schema_handler (SchemaHandler): An instance of SchemaHandler.
        """
        self.schema_handler = schema_handler
        self.logger = logging.getLogger(__name__)

    def process(self, data: dict) -> tuple:
        """
        Processes the given data to align it with the schema.

        Args:
            data (dict): The JSON response data.

        Returns:
            tuple: A tuple of (processed_data, unmatched_keys).
        """
        self.logger.debug("Starting heuristic processing for data: %s", data)

        schema = self.schema_handler.schema
        if not schema:
            self.logger.error("No schema provided in the SchemaHandler.")
            raise ValueError("No schema provided for processing.")

        processed = {}
        unmatched_keys = []

        # Check keys in the data to find extra keys
        for key, value in data.items():
            if key in schema and isinstance(value, schema[key]):
                processed[key] = value
                self.logger.debug(
                    "Matched key '%s' with value '%s' to the schema.", key, value
                )
            else:
                unmatched_keys.append(key)
                self.logger.debug("Unmatched key '%s' with value '%s'.", key, value)

        # Check for keys in the schema that are missing from the data
        for key in schema:
            if key not in data:
                unmatched_keys.append(key)
                self.logger.debug("Key '%s' in schema is missing from data.", key)

        self.logger.info(
            "Heuristic processing completed. Processed: %s, Unmatched keys: %s",
            processed,
            unmatched_keys,
        )

        return processed, unmatched_keys
