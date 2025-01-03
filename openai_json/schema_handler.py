import logging
import json
from jsonschema import Draft7Validator, validate, ValidationError, exceptions
import re


class SchemaNotSubmittedError(Exception):
    """
    Raised when a schema-dependent operation is called before submitting a schema.
    """

    pass


class SchemaHandler:
    """
    Manages schemas for JSON responses, including submission and validation.

    This class is responsible for accepting user-defined schemas, validating them,
    and ensuring that data conforms to the specified structure.

    Attributes:
        original_schema (dict): User-submitted schema with original keys.
        normalized_schema (dict): Normalized schema for processing.
        key_mapping (dict): Maps normalized keys back to their original forms.
    """

    python_type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        list: "list",
        bool: "boolean",
        dict: "object",
    }

    def __init__(self, schema=None):
        """
        Initializes the SchemaHandler with no schema loaded.
        """
        self.original_schema = None  # Keeps schema with original keys
        self.normalized_schema = None  # Keeps schema with normalized keys
        self.key_mapping = {}  # Map normalized keys to original keys
        self.logger = logging.getLogger(__name__)

        self.python_type_reverse_mapping = {
            v: k for k, v in self.python_type_mapping.items()
        }

        if schema:
            self.submit_schema(schema)

    def submit_schema(self, schema: dict):
        """
        Submits a new schema for validation and storage.

        Args:
            schema (dict): The schema to validate and store.

        Raises:
            ValueError: If the schema is invalid.
        """
        self.logger.info("Submitting a new schema for validation.")

        # Convert JSON string to dictionary if necessary
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
                self.logger.debug("Converted JSON string to dictionary: %s", schema)
            except json.JSONDecodeError as e:
                self.logger.error("Failed to decode JSON string: %s", e)
                raise ValueError(f"Invalid JSON string: {e}")

        # Ensure the schema is a dictionary
        if not isinstance(schema, dict):
            raise ValueError(
                f"Unsupported schema format: Expected a dictionary, got: {type(schema).__name__}"
            )

        # Validate the schema (without normalization)
        try:
            Draft7Validator.check_schema(schema)
            self.logger.info("Schema validated successfully.")
        except exceptions.SchemaError as e:
            self.logger.error("Invalid schema submitted: %s", e.message)
            raise ValueError(f"Invalid schema: {e.message}")

        self.logger.info("Schema submitted successfully.")

        # Store the original schema
        self.original_schema = schema

        # Normalize schema for Python-specific processing
        self.normalized_schema = self._normalize_schema(schema)

    def _ensure_schema_submitted(self):
        """
        Ensures that a schema has been submitted. Raises an error if not.

        Raises:
            SchemaNotSubmittedError: If no schema has been submitted.
        """
        if not self.normalized_schema:
            self.logger.error("Schema not submitted.")
            raise SchemaNotSubmittedError(
                "Schema must be submitted before calling this method."
            )

    def generate_example_json(self) -> str:
        """
        Generates an example JSON string based on the normalized schema.

        Returns:
            str: A JSON-formatted string representing an example output.
        """
        self._ensure_schema_submitted()

        example = {}
        for key, details in self.normalized_schema.items():
            field_type = details.get("type")
            # Handle nested fields
            if "." in key:
                parts = key.split(".")
                current = example
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = (
                    "example string"
                    if field_type == "string"
                    else (
                        []
                        if field_type in ["array", "list"]
                        else (
                            {}
                            if field_type == "object"
                            else (
                                123
                                if field_type == "integer"
                                else (
                                    123.45
                                    if field_type == "number"
                                    else True if field_type == "boolean" else None
                                )
                            )
                        )
                    )
                )
            else:
                # Handle non-nested fields
                if field_type == "string":
                    example[key] = "example string"
                elif field_type == "integer":
                    example[key] = 123
                elif field_type == "number":
                    example[key] = 123.45
                elif field_type == "boolean":
                    example[key] = True
                elif field_type in ["array", "list"]:
                    example[key] = []
                elif field_type == "object":
                    example[key] = {}
                else:
                    example[key] = None  # Default fallback for unknown types
        self.logger.debug("Generated example JSON: %s", example)
        return json.dumps(example, indent=2)

    def extract_prompts(
        self, prefix: str = "Here are the field-specific instructions:"
    ) -> str:
        """
        Extracts prompts from the schema and formats them into a string,
        returning keys in their original format with an optional prefix.

        Args:
            prefix (str): A default prefix to prepend to the extracted prompts.

        Returns:
            str: A formatted string of field prompts, prefixed by the provided string.
        """
        self._ensure_schema_submitted()

        self.logger.debug(
            "Starting prompt extraction. Normalized schema: %s", self.normalized_schema
        )

        prompts = []
        for normalized_key, definition in self.normalized_schema.items():
            # Check if this is a field definition
            if isinstance(definition, dict) and "prompt" in definition:
                original_key = self.get_original_key(
                    normalized_key
                )  # Map normalized to original key
                self.logger.debug(
                    "Mapped normalized key '%s' to original key '%s'",
                    normalized_key,
                    original_key,
                )
                prompts.append(f"{original_key}: {definition['prompt']}")
            else:
                self.logger.debug("No prompt found for key '%s'", normalized_key)

        if prompts:
            prompts.insert(0, prefix)  # Add the prefix at the beginning of the prompts

        self.logger.debug("Extracted prompts with prefix: %s", prompts)
        return "\n".join(prompts)

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
        self._ensure_schema_submitted()

        self.logger.debug("Validating data against the schema: %s", data)
        normalized_data = {self.normalize_text(k): v for k, v in data.items()}
        try:
            # Validate the data against the normalized schema
            validate(instance=normalized_data, schema=self.normalized_schema)
            self.logger.info("Data validation passed.")
            reconstructed_data = {
                self.key_mapping.get(k, k): v for k, v in normalized_data.items()
            }
            return True, reconstructed_data
        except ValidationError as e:
            self.logger.warning("Data validation failed. Error: %s", e.message)
            # Map normalized error path back to the original key
            error_path = ".".join(self.key_mapping.get(part, part) for part in e.path)
            return False, f"Validation failed: {error_path}: {e.message}"
        except Exception as e:
            self.logger.error("Unexpected error during validation: %s", str(e))
            return False, f"Unexpected validation error: {str(e)}"

    def get_original_key(self, normalized_key: str) -> str:
        """
        Retrieves the original key corresponding to a normalized key.

        Args:
            normalized_key (str): The normalized key.

        Returns:
            str: The original key or the normalized key if no mapping exists.
        """
        self._ensure_schema_submitted()
        return self.key_mapping.get(normalized_key, normalized_key)

    def get_type_from_field(self, field: dict or str):
        """
        Retrieves the expected Python type from a schema field definition.

        Args:
            field (dict or str): The field definition in the schema. Can be a dict with 
                a "type" key or a shorthand type string.

        Returns:
            type or None: The Python type corresponding to the schema field, or None 
                if the type is undefined.

        Example:
            python_type_reverse_mapping = {
            "string": str,
            "number": float,
            "integer": int
            }
            field = {"type": "string"}
            >>> schema_handler.get_type_from_field(field)
            <class 'str'>
        """


        self.logger.debug("Field definition passed to get_type_from_field: %s", field)

        if isinstance(field, dict) and "type" in field:
            json_type = field["type"]

            # Validate that the type is not nested or malformed
            if isinstance(json_type, dict):
                self.logger.error("Malformed 'type' field detected: %s", json_type)
                raise ValueError(f"Invalid nested 'type': {json_type}")

            # Map JSON type to Python type
            python_type = self.python_type_reverse_mapping.get(json_type)
            if not python_type:
                self.logger.warning("Unknown type '%s' in field definition.", json_type)
            return python_type

        elif isinstance(field, str):
            # Handle shorthand type strings
            return self.python_type_reverse_mapping.get(field)

        # Undefined or unsupported field type
        self.logger.warning("Unsupported field definition format: %s", field)
        return None

    def get_field_expected_type(self, key: str):
        """
        Retrieves the expected Python type for a given field key based on the schema.

        Args:
            key (str): The normalized key of the field.

        Returns:
            type or None: The Python type corresponding to the field, or None if undefined.

        Example:
            schema = {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "tags": {"type": "list", "items": {"type": "string"}}
            }
            >>> schema_handler.get_field_expected_type("name")
            <class 'str'>
            >>> schema_handler.get_field_expected_type("tags.items")
            <class 'str'>
        """
        self._ensure_schema_submitted()
        # Check if the key directly exists in the schema
        field_definition = self.normalized_schema.get(key)
        if field_definition:
            # Use get_type_from_field to resolve the type
            return self.get_type_from_field(field_definition)

        # Handle 'items' key for list schemas (e.g., 'key_a.items')
        if "." in key and key.endswith(".items"):
            parent_key = key.rsplit(".", 1)[0]  # Extract parent key (e.g., 'key_a')
            parent_field = self.normalized_schema.get(parent_key)
            if parent_field and parent_field.get("type") == "list":
                items_definition = parent_field.get("items")
                if items_definition:
                    return self.get_type_from_field(items_definition)

        # Log and return None if the field is not found or cannot be resolved
        self.logger.debug("Field '%s' not found in schema.", key)
        return None

    def register_type(self, python_type: type, json_type: str):
        """
        Registers a custom Python-to-JSON type mapping.

        This method allows the user to extend the default type mappings used in the
        schema handler. It ensures compatibility between Python types and their
        corresponding JSON Schema types.

        Args:
            python_type (type): The Python type to map (e.g., `datetime`, `Decimal`).
            json_type (str): The JSON Schema type to associate with the Python type
                (e.g., "string", "number").

        Raises:
            ValueError: If `python_type` is not a valid type or `json_type` is not a string.

        Example:
            >>> handler = SchemaHandler()
            >>> handler.register_type(datetime, "string")
            >>> print(handler.python_type_mapping)
            {str: "string", int: "integer", ..., datetime: "string"}
        """
        if not isinstance(python_type, type) or not isinstance(json_type, str):
            raise ValueError("Invalid type mapping. Expected (type, str).")
        self.python_type_mapping[python_type] = json_type
        self.python_type_reverse_mapping[json_type] = python_type
        self.logger.info(
            "Registered custom type mapping: %s -> %s", python_type, json_type
        )

    def add_field(self, field_name: str, field_schema: dict):
        """
        Dynamically adds a new field to the schema.

        This method allows users to extend the schema dynamically by adding new
        fields. The field schema must conform to JSON Schema standards and is
        normalized before being added.

        Args:
            field_name (str): The name of the field to add.
            field_schema (dict): The schema definition of the field (e.g., `{"type": "string"}`).

        Raises:
            ValueError: If the schema does not conform to JSON Schema standards or if
                `field_name` is invalid.

        Example:
            >>> handler = SchemaHandler()
            >>> handler.submit_schema({"type": "object", "properties": {}})
            >>> handler.add_field("new_field", {"type": "integer"})
            >>> print(handler.normalized_schema)
            {"type": "object", "properties": {"new_field": {"type": "integer"}}}
        """
        self._ensure_schema_submitted()
        if not isinstance(field_name, str) or not isinstance(field_schema, dict):
            raise ValueError("Invalid field name or schema. Expected (str, dict).")
        normalized_key = self.normalize_text(field_name)
        self.key_mapping[normalized_key] = field_name
        normalized_field = self._normalize_field(field_schema)
        self.normalized_schema["properties"][normalized_key] = normalized_field
        self.logger.info("Added field '%s' to the schema.", field_name)

    def diff_schema(self, new_schema: dict) -> dict:
        """
        Compares the current schema with a new schema and identifies differences.

        This method performs a deep comparison of the original schema and a new
        schema. It highlights added, removed, and changed fields to assist in
        tracking schema updates.

        Args:
            new_schema (dict): The new schema to compare against the current schema.

        Returns:
            dict: A dictionary with the following keys:
                - `added`: Fields present in the new schema but not in the current schema.
                - `removed`: Fields present in the current schema but not in the new schema.
                - `changed`: Fields present in both schemas but with different definitions.

        Raises:
            ValueError: If no schema has been submitted yet or if `new_schema` is not a dictionary.

        Example:
            >>> handler = SchemaHandler()
            >>> handler.submit_schema({"type": "object", "properties": {"field1": {"type": "string"}}})
            >>> new_schema = {"type": "object", "properties": {"field1": {"type": "integer"}, "field2": {"type": "boolean"}}}
            >>> diff = handler.diff_schema(new_schema)
            >>> print(diff)
            {
                "added": {"field2": {"type": "boolean"}},
                "removed": {},
                "changed": {"field1": ({"type": "string"}, {"type": "integer"})}
            }
        """
        self._ensure_schema_submitted()

        if not isinstance(new_schema, dict):
            raise ValueError("Invalid schema format. Expected a dictionary.")

        added = {
            k: v
            for k, v in new_schema.get("properties", {}).items()
            if k not in self.original_schema.get("properties", {})
        }
        removed = {
            k: v
            for k, v in self.original_schema.get("properties", {}).items()
            if k not in new_schema.get("properties", {})
        }
        changed = {
            k: (self.original_schema["properties"][k], new_schema["properties"][k])
            for k in new_schema.get("properties", {})
            if k in self.original_schema.get("properties", {})
            and self.original_schema["properties"][k] != new_schema["properties"][k]
        }

        return {"added": added, "removed": removed, "changed": changed}

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalizes a given text by:
        - Converting CamelCase to spaced words.
        - Replacing underscores, dashes, and other delimiters with spaces.
        - Removing parenthetical phrases.
        - Handling variations of conjunctions like "and", "&", "/", "and/or".
        - Converting spaces to underscores.
        - Converting to lowercase.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        # Replace underscores, dashes, and slashes with spaces
        text = re.sub(r"[_\-/]", " ", text)
        # Insert a space before capital letters (for CamelCase)
        text = re.sub(r"(?<=[a-z])([A-Z])", r" \1", text)
        # Remove parenthetical phrases
        text = re.sub(r"\([^)]*\)", "", text)
        # Normalize conjunction variations: "and", "&", "/", "and/or"
        text = re.sub(r"\b(and/or|&|/)\b", " and ", text, flags=re.IGNORECASE)
        # Normalize extra spaces and convert to lowercase
        text = " ".join(text.lower().split())
        # Replace spaces with underscores
        return text.replace(" ", "_")

    def map_keys_to_original(self, data: dict) -> dict:
        """
        Maps normalized keys to original keys using the key_mapping.

        Args:
            data (dict): Data with normalized keys.

        Returns:
            dict: Data with keys mapped back to their original forms.
        """
        self._ensure_schema_submitted()

        def transform_field(field):
            if isinstance(field, dict):
                if field.get("type") == "list":
                    field["type"] = "array"  # Convert back for JSON Schema compliance
            return field

        return {
            self.key_mapping.get(key, key): (
                self.map_keys_to_original(value)
                if isinstance(value, dict)
                else transform_field(value)
            )
            for key, value in data.items()
        }

    def _normalize_schema(self, schema: dict) -> dict:
        """
        Normalizes the schema to ensure consistent structure.
        """
        normalized_schema = {}

        for key, value in schema.items():
            normalized_key = self.normalize_text(key)
            if normalized_key in self.key_mapping:
                self.logger.warning(
                    "Normalization conflict: Original keys '%s' and '%s' normalize to the same value.",
                    key,
                    self.key_mapping[normalized_key],
                )
            self.key_mapping[normalized_key] = key  # Store mapping

            if key == "properties":
                if not isinstance(value, dict):
                    self.logger.error(
                        "Invalid schema format for 'properties': %s", value
                    )
                    raise ValueError(f"Invalid schema format for 'properties': {value}")
                normalized_schema[normalized_key] = {
                    self.normalize_text(sub_key): self._normalize_field(sub_value)
                    for sub_key, sub_value in value.items()
                }
            elif key == "required":
                if not isinstance(value, list):
                    self.logger.error("Invalid schema format for 'required': %s", value)
                    raise ValueError(f"Invalid schema format for 'required': {value}")
                normalized_schema[normalized_key] = [
                    self.normalize_text(k) for k in value
                ]
            elif key in ["additionalProperties", "type"]:
                normalized_schema[normalized_key] = value
            else:
                normalized_schema[normalized_key] = self._normalize_field(value)

        self.logger.debug("Schema normalization completed: %s", normalized_schema)
        return normalized_schema

    def _normalize_field(self, field: dict or str or type) -> dict:
        """
        Normalizes an individual field in the schema.
        """
        if isinstance(field, str):  # Simplified format, e.g., "integer"
            self.logger.debug("Normalizing simplified field: %s", field)
            return {
                "type": field if field != "array" else "list"
            }  # Convert array to list
        elif isinstance(field, dict):  # Detailed format, e.g., {"type": "integer"}
            self.logger.debug("Normalizing detailed field: %s", field)
            if field.get("type") == "array":  # Convert array to list
                field["type"] = "list"
            return field
        elif isinstance(field, type):  # Python type, e.g., str
            json_type = self.python_type_mapping.get(field)
            if not json_type:
                self.logger.error("Unsupported Python type in schema: %s", field)
                raise ValueError(f"Unsupported Python type in schema: {field}")
            self.logger.debug("Normalizing Python type field: %s", field)
            return {"type": json_type}
        else:
            self.logger.error("Invalid field format: %s", field)
            raise ValueError(f"Invalid field format: {field}")
