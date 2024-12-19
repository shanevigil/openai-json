class SubstructureManager:
    """
    Manages unmatched keys or substructures from JSON responses.

    This class provides functionality to store, retrieve, and clear data
    corresponding to keys that do not conform to a specified schema.

    Attributes:
        unmatched_data (dict): A dictionary containing unmatched keys and their associated data.
    """

    def __init__(self):
        """
        Initializes the SubstructureManager with an empty unmatched data store.
        """
        self.unmatched_data = {}

    def store_unmatched_keys(self, unmatched_keys: list, data: dict):
        """
        Stores keys and their corresponding data that don't match the schema.

        Args:
            unmatched_keys (list): List of keys not matched by the schema.
            data (dict): The original response data containing unmatched keys.
        """
        for key in unmatched_keys:
            if key in data:
                self.unmatched_data[key] = data[key]

    def retrieve_unmatched_data(self) -> dict:
        """
        Retrieves all unmatched data stored in the manager.

        Returns:
            dict: A dictionary of unmatched keys and their associated data.
        """
        return self.unmatched_data

    def clear(self):
        """
        Clears all stored unmatched data.
        """
        self.unmatched_data.clear()
