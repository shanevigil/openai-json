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

    def submit_schema(self, schema: dict):
        """
        Submits a new schema for validation and storage.

        Args:
            schema (dict): The schema to validate and store.

        Raises:
            ValueError: If the schema is invalid.
        """
        try:
            # Validate the schema itself
            Draft7Validator.check_schema(schema)
            self.schema = schema
        except exceptions.SchemaError as e:
            raise ValueError(f"Invalid schema: {e.message}")

    def validate_data(self, data: dict) -> bool:
        """
        Validates the given data against the current schema.

        Args:
            data (dict): The data to validate.

        Returns:
            bool: True if the data conforms to the schema, False otherwise.

        Raises:
            ValueError: If no schema has been submitted.
        """
        if not self.schema:
            raise ValueError("No schema has been submitted.")
        try:
            # Validate the data against the schema
            validate(instance=data, schema=self.schema)
            return True, "Validation passed."
        except ValidationError as e:
            return False, f"Validation failed: {e.message}"
