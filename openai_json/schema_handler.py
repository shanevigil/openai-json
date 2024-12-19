import logging
from jsonschema import Draft7Validator, validate, ValidationError, exceptions


class SchemaHandler:
    """
    Manages schemas for JSON responses, including submission and validation.

    This class is responsible for accepting user-defined schemas, validating them,
    and ensuring that data conforms to the specified structure.

    Attributes:
        schema (dict): The current schema used for validation.
    """

    def __init__(self):
        """
        Initializes the SchemaHandler with no schema loaded.
        """
        self.schema = None  # Stores the current schema
        self.logger = logging.getLogger(__name__)

    def submit_schema(self, schema: dict):
        """
        Submits a new schema for validation and storage.

        Args:
            schema (dict): The schema to validate and store.

        Raises:
            ValueError: If the schema is invalid.
        """
        self.logger.info("Submitting a new schema for validation.")
        try:
            # Validate the schema itself
            Draft7Validator.check_schema(schema)
            self.schema = schema
            self.logger.info("Schema submitted and validated successfully.")
        except exceptions.SchemaError as e:
            self.logger.error("Invalid schema submitted: %s", e.message)
            raise ValueError(f"Invalid schema: {e.message}")

    def validate_data(self, data: dict) -> tuple:
        """
        Validates the given data against the current schema.

        Args:
            data (dict): The data to validate.

        Returns:
            tuple: A tuple containing a boolean indicating validation success
                   and a message describing the validation result.

        Raises:
            ValueError: If no schema has been submitted.
        """
        if not self.schema:
            self.logger.error("No schema has been submitted. Cannot validate data.")
            raise ValueError("No schema has been submitted.")

        self.logger.debug("Validating data against the schema: %s", data)
        try:
            # Validate the data against the schema
            validate(instance=data, schema=self.schema)
            self.logger.info("Data validation passed.")
            return True, "Validation passed."
        except ValidationError as e:
            self.logger.warning(
                "Data validation failed. Error: %s. Data: %s", e.message, data
            )
            return False, f"Validation failed: {e.message}"
