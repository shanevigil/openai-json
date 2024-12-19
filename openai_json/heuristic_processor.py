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

    def process(self, data: dict) -> tuple:
        """
        Processes the given data to align it with the schema.

        Args:
            data (dict): The JSON response data.

        Returns:
            tuple: A tuple of (processed_data, unmatched_keys).
        """
        schema = self.schema_handler.schema
        processed = {}
        unmatched_keys = []

        # Check keys in the data to find extra keys
        for key, value in data.items():
            if key in schema and isinstance(value, schema[key]):
                processed[key] = value
            else:
                unmatched_keys.append(key)

        # Check for keys in the schema that are missing from the data
        for key in schema:
            if key not in data:
                unmatched_keys.append(key)

        return processed, unmatched_keys
