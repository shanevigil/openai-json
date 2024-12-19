class HeuristicProcessor:
    """Applies heuristic rules to align JSON data with the schema."""

    def __init__(self, schema_handler):
        self.schema_handler = schema_handler

    def process(self, data: dict):
        """Matches JSON data with the schema."""
        schema = self.schema_handler.schema
        processed = {}
        unmatched_keys = []

        for key, expected_type in schema.items():
            if key in data and isinstance(data[key], expected_type):
                processed[key] = data[key]
            else:
                unmatched_keys.append(key)

        return processed, unmatched_keys
