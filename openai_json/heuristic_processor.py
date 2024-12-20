import logging
from openai_json.schema_handler import SchemaHandler


class HeuristicProcessor:
    """
    Applies heuristic rules to process and match JSON structures to schemas.

    This class is responsible for aligning straightforward elements of JSON
    responses to their corresponding schema using predefined rules.

    Attributes:
        schema_handler (SchemaHandler): An instance of SchemaHandler for schema validation and type mapping.
    """

    def __init__(self, schema_handler: SchemaHandler):
        """
        Initializes the HeuristicProcessor with a schema handler.

        Args:
            schema_handler (SchemaHandler): An instance of SchemaHandler.
        """
        self.schema_handler = schema_handler
        self.logger = logging.getLogger(__name__)

    def process(self, data: dict) -> tuple:
        """
        Processes the given data to align it with the schema.

        Args:
            data (dict): The JSON response data.

        Returns:
            tuple: A tuple of (processed_data, unmatched_keys).
        """
        self.logger.debug("Starting heuristic processing for data: %s", data)

        schema = self.schema_handler.normalized_schema
        if not schema:
            self.logger.error("No schema provided in SchemaHandler.")
            raise ValueError("No schema provided for processing.")

        # Delegate processing to the nested method for recursion
        processed, unmatched_keys = self._process_nested(data, schema)

        return processed, unmatched_keys

    def _process_nested(self, data: dict, schema: dict, path: str = "") -> tuple:
        """
        Recursively processes nested structures in the data to align them with the schema.

        Args:
            data (dict): The JSON response data.
            schema (dict): The schema definition.
            path (str): The current path in the nested structure.

        Returns:
            tuple: A tuple of (processed_data, unmatched_keys).
        """
        processed = {}
        unmatched_keys = []
        normalized_data = {
            self.schema_handler.normalize_text(key): value
            for key, value in data.items()
        }

        self.logger.debug("Processing nested data at path '%s': %s", path, data)

        for key, value in normalized_data.items():
            current_path = f"{path}.{key}" if path else key
            normalized_key = self.schema_handler.normalize_text(key)
            field_definition = schema.get("properties", {}).get(
                normalized_key, schema.get(normalized_key)
            )
            expected_type = self.schema_handler.get_type_from_field(field_definition)

            self.logger.debug(
                "Key: '%s', Normalized Key: '%s', Value: '%s', Field Definition: %s, Expected Type: %s",
                key,
                normalized_key,
                value,
                field_definition,
                expected_type,
            )

            if expected_type is list:
                self.logger.debug("Value of key '%s' is expected to be a List", key)
                list_processed, list_unmatched = self._process_list(
                    value, field_definition, current_path
                )
                if list_processed:
                    processed[normalized_key] = list_processed
                unmatched_keys.extend(list_unmatched)

            elif expected_type is dict and isinstance(value, dict):
                self.logger.debug("Value of key '%s' is of type Dict", key)
                nested_schema = field_definition.get("properties", field_definition)
                nested_processed, nested_unmatched = self._process_nested(
                    value, nested_schema, current_path
                )
                if nested_processed:
                    processed[normalized_key] = nested_processed
                unmatched_keys.extend(nested_unmatched)

            elif expected_type and isinstance(value, expected_type):
                self.logger.debug(
                    "Value of key '%s' is of expected type %s", key, expected_type
                )
                processed[normalized_key] = value
                self.logger.debug(
                    "Matched key '%s' with value '%s' to the schema.",
                    current_path,
                    value,
                )

            else:
                self.logger.debug(
                    "The '%s' contains unexpected nesting or type mismatch. Value is: %s",
                    key,
                    value,
                )
                unmatched_keys.append(normalized_key)  # Append just the key name

        self.logger.debug(
            "Completed processing for path '%s'. Processed: %s, Unmatched: %s",
            path,
            processed,
            unmatched_keys,
        )

        return processed, unmatched_keys

    def _process_dict(
        self, value: dict, field_definition: dict, current_path: str
    ) -> tuple:
        if "properties" in field_definition:
            nested_schema = field_definition["properties"]
            return self._process_nested(value, nested_schema, current_path)
        else:
            self.logger.error(
                "Invalid schema for nested object at path '%s'.", current_path
            )
            return {}, [current_path]

    def _process_list(
        self, value: any, field_definition: dict, current_path: str
    ) -> tuple:
        """
        Processes a list value against the schema.

        Args:
            value (list): The list data to process.
            field_definition (dict): The schema definition for the list.
            current_path (str): The current path in the data.

        Returns:
            tuple: Processed list and unmatched keys/items.
        """
        matched_items = []
        unmatched_items = {}

        # Handle string inputs if the schema expects a list
        if isinstance(value, str) and field_definition.get("type") == "list":
            self.logger.debug(
                "Value for '%s' is a string but expected a list. Attempting to parse.",
                current_path,
            )
            value = [item.strip() for item in value.split(",")]

        if not isinstance(value, list):
            # If the value still isn't a list, it is unmatched
            self.logger.debug(
                "Value for '%s' is not a list after parsing. Treating as unmatched.",
                current_path,
            )
            return [], [{current_path: value}]

        if "items" in field_definition:
            item_schema = field_definition["items"]
            self.logger.debug(
                "Processing list items at path '%s' with schema: %s",
                current_path,
                item_schema,
            )

            for index, item in enumerate(value):
                item_path = f"{current_path}[{index}]"

                if isinstance(item, dict):
                    # Recurse for nested objects in the list
                    nested_processed, nested_unmatched = self._process_nested(
                        item, item_schema, item_path
                    )
                    if nested_processed:
                        matched_items.append(nested_processed)
                    if nested_unmatched:
                        unmatched_items.update({item_path: nested_unmatched})
                else:
                    # Validate primitive list items
                    item_type = self.schema_handler.get_type_from_field(item_schema)
                    if item_type and isinstance(item, item_type):
                        matched_items.append(item)
                        self.logger.debug(
                            "Matched list item at '%s' with value '%s' to the schema.",
                            item_path,
                            item,
                        )
                    else:
                        unmatched_items[item_path] = [item]
                        self.logger.debug(
                            "Unmatched list item at '%s' with value '%s' (type mismatch or not in schema).",
                            item_path,
                            item,
                        )
        else:
            # Default behavior for lists without 'items'
            self.logger.warning(
                "Schema for list '%s' does not define 'items'. Assuming all list items are valid.",
                current_path,
            )
            matched_items = (
                value  # Assume all items in the list are valid as no schema is provided
            )

        # Return unmatched items as a flat dictionary for consistency with test expectations
        unmatched_list_keys = [unmatched_items] if unmatched_items else []
        return matched_items, unmatched_list_keys

    def _process_primitive(
        self, processed: dict, key: str, value: any, current_path: str
    ):
        processed[key] = value
        self.logger.debug(
            "Matched key '%s' with value '%s' to the schema.", current_path, value
        )
