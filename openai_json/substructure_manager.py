import logging


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
        self.logger = logging.getLogger(__name__)

    def store_unmatched_keys(self, unmatched_keys: list, data: dict):
        """
        Stores keys and their corresponding data that don't match the schema.

        Args:
            unmatched_keys (list): List of keys not matched by the schema.
            data (dict): The original response data containing unmatched keys.
        """
        self.logger.info("Storing unmatched keys: %s", unmatched_keys)
        for key in unmatched_keys:
            if key in data:
                self.unmatched_data[key] = data[key]
                self.logger.debug(
                    "Stored unmatched key '%s' with value '%s'.", key, data[key]
                )
            else:
                self.logger.warning(
                    "Unmatched key '%s' not found in the provided data.", key
                )

    def retrieve_unmatched_data(self) -> dict:
        """
        Retrieves all unmatched data stored in the manager.

        Returns:
            dict: A dictionary of unmatched keys and their associated data.
        """
        self.logger.debug("Retrieving unmatched data: %s", self.unmatched_data)
        return self.unmatched_data

    def clear(self):
        """
        Clears all stored unmatched data.
        """
        self.logger.info("Clearing all unmatched data.")
        self.unmatched_data.clear()
        self.logger.debug("Unmatched data cleared successfully.")
