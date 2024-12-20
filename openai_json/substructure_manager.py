import logging


class SubstructureManager:
    """
    Manages unmatched keys or substructures from JSON responses.

    This class provides functionality to store, retrieve, and clear data
    corresponding to keys that do not conform to a specified schema.

    Attributes:
        unmatched_data (dict): A dictionary containing unmatched keys and their associated data.
    """

    def __init__(self, schema_handler):
        """
        Initializes the SubstructureManager with an empty unmatched data store.

        Args:
            schema_handler (SchemaHandler): Instance of SchemaHandler for normalization and schema utilities.
        """
        self.unmatched_keys = {}
        self.schema_handler = schema_handler
        self.logger = logging.getLogger(__name__)

    def store_unmatched_keys(self, keys, data):
        """
        Stores unmatched keys and their values from the provided data.
        Args:
            keys (list): List of unmatched keys, which may include nested keys (e.g., "level1.level2.key2").
            data (dict): The data to extract values for unmatched keys.
        """
        self.logger.info("Storing unmatched keys: %s", keys)

        for key in keys:
            normalized_key = self.schema_handler.normalize_text(key)

            # Split nested keys and traverse the data
            key_parts = normalized_key.split(".")
            value = self._get_nested_value(key_parts, data)

            if value is not None:
                self.unmatched_keys[normalized_key] = value
                self.logger.debug(
                    "Stored unmatched key '%s' with value '%s'.",
                    normalized_key,
                    value,
                )
            else:
                self.logger.warning(
                    "Unmatched key '%s' not found in the provided data.", key
                )

    def _get_nested_value(self, key_parts, data):
        """
        Recursively retrieves the value for a nested key.

        Args:
            key_parts (list): List of key parts representing the nested path (e.g., ["level1", "level2", "key2"]).
            data (dict): The data to search.

        Returns:
            The value if the key is found, otherwise None.
        """
        if not key_parts or not isinstance(data, dict):
            return None

        current_key = key_parts[0]
        if current_key in data:
            if len(key_parts) == 1:  # Base case: final key part
                return data[current_key]
            return self._get_nested_value(key_parts[1:], data[current_key])
        return None

    def retrieve_unmatched_data(self):
        """
        Retrieves all stored unmatched data.

        Returns:
            dict: A dictionary of unmatched keys and their values.
        """
        self.logger.debug("Retrieving unmatched data: %s", self.unmatched_keys)
        return self.unmatched_keys

    def _find_key_in_data(self, key, data):
        """
        Recursively searches for a key in the data.

        Args:
            key (str): The key to find (assumed normalized).
            data (dict): The data to search in.

        Returns:
            The value if the key is found, otherwise None.
        """
        if key in data:
            return data[key]

        # Check nested dictionaries
        for sub_key, sub_value in data.items():
            if isinstance(sub_value, dict):
                result = self._find_key_in_data(key, sub_value)
                if result is not None:
                    return result

        return None
