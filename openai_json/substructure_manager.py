class SubstructureManager:
    """Manages unmatched keys or structures for further processing."""

    def __init__(self):
        self.unmatched_data = {}

    def store_unmatched_keys(self, unmatched_keys: list, data: dict):
        """
        Stores keys and their corresponding data that don't match the schema.
        """
        for key in unmatched_keys:
            if key in data:
                self.unmatched_data[key] = data[key]

    def retrieve_unmatched_data(self):
        """
        Retrieves all unmatched data.
        """
        return self.unmatched_data
