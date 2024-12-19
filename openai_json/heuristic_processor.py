class HeuristicProcessor:
    """Applies heuristic rules to align JSON data with the schema."""

    def __init__(self, schema_handler):
        self.schema_handler = schema_handler

    def process(self, data: dict):
        """Matches JSON data with the schema."""
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
