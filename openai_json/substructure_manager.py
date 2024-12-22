import logging


class SubstructureManager:
    """
    Manages substructures, including unmatched data and errors, for a response processing pipeline.

    Attributes:
        schema_handler (SchemaHandler): The schema handler used for key normalization and validation.
        unmatched_data_store (dict): Stores unmatched data from the response.
        error_data_store (dict): Stores errors identified during heuristic processing.
    """

    def __init__(self, schema_handler):
        """
        Initializes the SubstructureManager with a schema handler.

        Args:
            schema_handler (SchemaHandler): A schema handler instance to assist with key normalization.
        """
        self.schema_handler = schema_handler
        self.unmatched_data_store = []
        self.error_data_store = []
        self.logger = logging.getLogger(__name__)

    def store_unmatched_data(self, unmatched_data):
        """
        Stores unmatched data in the manager, preserving normalized keys for subsequent processing.

        Args:
            unmatched_data (dict): A dictionary of normalized keys from heuristic processing.

        Side Effects:
            Updates the unmatched_data_store with normalized keys and corresponding values.
        """
        for item in unmatched_data:
            self.unmatched_data_store.append(item)

    def retrieve_unmatched_data(self):
        """
        Retrieves normalized unmatched data stored in the manager.

        Returns:
            dict: The unmatched data with normalized keys and their corresponding values.
        """
        return self.unmatched_data_store

    def store_error_data(self, errors):
        """
        Stores error data in the manager by normalizing paths and associating them with their values.

        Args:
            errors (list): A list of dictionaries where each entry represents an error with a path and value.

        Side Effects:
            Updates the error_data_store with normalized paths and their corresponding values.
        """
        for error in errors:
            self.error_data_store.append(error)

    def retrieve_error_data(self):
        """
        Retrieves error data stored in the manager.

        Returns:
            dict: The error data with normalized paths and their corresponding values.
        """
        return self.error_data_store

    def reconcile_transformed_data(self, transformed_data):
        """
        Updates unmatched data with transformed values and removes reconciled keys.

        Args:
            transformed_data (list[dict] or dict): Transformed data returned by the ML processor.
        """
        self.logger.info("Reconciling transformed data.")
        self.logger.debug("Transformed data: %s", transformed_data)

        # Normalize transformed_data to a list of dictionaries
        if isinstance(transformed_data, dict):
            transformed_data = [transformed_data]

        keys_to_remove = []  # List of indices to remove from unmatched_data_store

        # If there's nothing to do, do nothing
        if not transformed_data:
            self.logger.debug(
                "No transformation data to reconcile: exiting reconciler."
            )
            return

        for transformed_dict in transformed_data:
            if not isinstance(transformed_dict, dict):
                self.logger.error(
                    "Invalid transformed data format: Expected dict, got %s",
                    type(transformed_dict),
                )
                continue

            for key, value in transformed_dict.items():
                found = False
                for i, unmatched_dict in enumerate(self.unmatched_data_store):
                    if key in unmatched_dict:
                        self.logger.debug(
                            "Reconciled unmatched data key '%s' with transformed value '%s'.",
                            key,
                            value,
                        )
                        keys_to_remove.append(i)  # Mark the index for removal
                        found = True
                        break
                if not found:
                    self.logger.warning(
                        "Transformed data key '%s' is not in unmatched data.", key
                    )

        # Remove the reconciled keys from unmatched_data_store
        for index in sorted(keys_to_remove, reverse=True):
            self.logger.debug(
                "Removing reconciled unmatched data at index '%s'.", index
            )
            del self.unmatched_data_store[index]

    def _retrieve_value_from_path(self, data, path):
        """
        Helper function to retrieve a value from a nested dictionary using a dot-separated path.

        Args:
            data (dict): The dictionary to search.
            path (str): The dot-separated path representing the hierarchy of keys.

        Returns:
            any: The value at the specified path if found, or None if any key in the path is missing.
        """
        keys = path.split(".")
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return None
        return data
