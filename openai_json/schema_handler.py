from jsonschema import Draft7Validator, validate, ValidationError, exceptions


class SchemaHandler:
    """Manages user-defined schemas and validates JSON data against them."""

    def __init__(self):
        self.schema = None  # Stores the current schema

    def submit_schema(self, schema: dict):
        """Accepts and stores a new schema."""
        try:
            # Validate the schema itself
            Draft7Validator.check_schema(schema)
            self.schema = schema
        except exceptions.SchemaError as e:
            raise ValueError(f"Invalid schema: {e.message}")

    def validate_data(self, data: dict):
        """Validates the provided data against the stored schema."""
        if not self.schema:
            raise ValueError("No schema has been submitted.")
        try:
            # Validate the data against the schema
            validate(instance=data, schema=self.schema)
            return True, "Validation passed."
        except ValidationError as e:
            return False, f"Validation failed: {e.message}"
