import logging
from openai_json.schema_handler import SchemaHandler
from word2number import w2n


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
            tuple: A tuple of (processed_data, unmatched_keys, errors).
        """
        self.logger.debug("Starting heuristic processing for data: %s", data)

        schema = self.schema_handler.normalized_schema
        if not schema:
            self.logger.error("No schema provided in SchemaHandler.")
            raise ValueError("No schema provided for processing.")

        processed, unmatched_keys, errors = self._process_nested(data, schema)

        return processed, unmatched_keys, errors

    def _process_nested(self, data: dict, schema: dict, path: str = "") -> tuple:
        """
        Recursively processes nested structures in the data.

        Args:
            data (dict): The JSON response data.
            schema (dict): The schema definition.
            path (str): The current path in the nested structure.

        Returns:
            tuple: A tuple of (processed_data, unmatched_keys).
        """
        processed = {}
        unmatched_keys = []
        errors = []
        normalized_data = self._normalize_data(data)

        for key, value in normalized_data.items():
            current_path = self._build_path(path, key)
            normalized_key = self.schema_handler.normalize_text(key)
            field_definition = self._get_field_definition(schema, normalized_key)
            expected_type = self.schema_handler.get_type_from_field(field_definition)

            if expected_type is list:
                self._process_list_field(
                    processed,
                    unmatched_keys,
                    errors,
                    key,
                    value,
                    field_definition,
                    current_path,
                )
            elif expected_type is dict and isinstance(value, dict):
                self._process_dict_field(
                    processed,
                    unmatched_keys,
                    errors,
                    key,
                    value,
                    field_definition,
                    current_path,
                )
            elif expected_type:
                self._process_primitive_field(
                    processed,
                    unmatched_keys,
                    errors,
                    key,
                    value,
                    normalized_key,
                    expected_type,
                    current_path,
                )
            else:
                self._process_unexpected_field(
                    processed, unmatched_keys, errors, key, value, schema, current_path
                )

        return processed, unmatched_keys, errors

    def _normalize_data(self, data: dict) -> dict:
        return {
            self.schema_handler.normalize_text(key): value
            for key, value in data.items()
        }

    def _build_path(self, path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    def _get_field_definition(self, schema: dict, normalized_key: str):
        return schema.get("properties", {}).get(
            normalized_key, schema.get(normalized_key)
        )

    def _process_list_field(
        self, processed, unmatched_keys, errors, key, value, field_definition, path
    ):
        list_processed, list_unmatched, list_errors = self._process_list(
            value, field_definition, path
        )
        if list_processed:
            processed[self.schema_handler.normalize_text(key)] = list_processed
        unmatched_keys.extend(list_unmatched)
        errors.extend(list_errors)

    def _process_dict_field(
        self, processed, unmatched_keys, errors, key, value, field_definition, path
    ):
        nested_schema = field_definition.get("properties", field_definition)
        nested_processed, nested_unmatched, nested_errors = self._process_nested(
            value, nested_schema, path
        )
        if nested_processed:
            processed[self.schema_handler.normalize_text(key)] = nested_processed
        unmatched_keys.extend(nested_unmatched)
        errors.extend(nested_errors)

    def _process_primitive_field(
        self,
        processed,
        unmatched_keys,
        errors,
        key,
        value,
        normalized_key,
        expected_type,
        path,
    ):
        try:
            coerced_value = self._coerce_item_type(value, expected_type)
            if isinstance(coerced_value, expected_type) or (
                expected_type == "number" and isinstance(coerced_value, (int, float))
            ):
                processed[normalized_key] = coerced_value
                self.logger.debug(
                    "Primitive processing succeeded - path: %s, key: %s, value: %s, coerced to: %s",
                    path,
                    key,
                    value,
                    coerced_value,
                )
            else:
                errors.append({path: value})
                self.logger.warning(
                    "Failed to coerce key: %s, value: %s, expected type: %s",
                    key,
                    value,
                    expected_type,
                )
        except (ValueError, TypeError) as e:
            self.logger.error(
                "Failed to process key '%s' at path '%s': %s", key, path, str(e)
            )
            errors.append({path: [value]})

    def _process_unexpected_field(
        self, processed, unmatched_keys, errors, key, value, schema, path
    ):
        if isinstance(value, dict):
            nested_processed, nested_unmatched, nested_errors = self._process_nested(
                value, schema, path
            )
            processed.update(nested_processed)
            unmatched_keys.extend(nested_unmatched)
            errors.extend(nested_errors)
        else:
            unmatched_keys.append({path: value})

    def _normalize_list_value(self, value, field_definition):
        if isinstance(value, str) and field_definition.get("type") == "list":
            return [item.strip() for item in value.split(",")]
        return value

    def _process_list(
        self, value: any, field_definition: dict, current_path: str
    ) -> tuple:
        value = self._normalize_list_value(value, field_definition)
        if not isinstance(value, list):
            return [], [{current_path: value}]

        matched_items, unmatched_items, errors = self._process_list_items(
            value, field_definition, current_path
        )

        return matched_items, unmatched_items, errors

    def _process_list_items(self, value, field_definition, current_path):
        matched_items = []
        unmatched_items = {}
        errors = []

        # Extract the expected schema for list items
        items_definition = field_definition.get("items", {})
        expected_type = self.schema_handler.get_type_from_field(items_definition)
        self.logger.debug(
            "Starting _process_list_items with value: %s, field_definition: %s, items_definition: %s, current_path: %s",
            value,
            field_definition,
            items_definition,
            current_path,
        )

        # If no explicit type is specified, infer the type from the first item
        if not expected_type and value:
            inferred_type = type(value[0])
            self.logger.debug(
                "No explicit type defined for list. Inferred type from first item: %s",
                inferred_type,
            )
            expected_type = inferred_type

        # Iterate over the list items
        for index, item in enumerate(value):
            item_path = f"{current_path}[{index}]"
            self.logger.debug("Processing list item at index %d: %s", index, item)
            try:
                if isinstance(item, dict):
                    self.logger.debug(
                        "Item is a dictionary. Field definition: %s, Items definition: %s",
                        field_definition,
                        items_definition,
                    )
                    # Process the nested dictionary using the item schema
                    nested_processed, nested_unmatched, nested_errors = (
                        self._process_nested(item, items_definition, item_path)
                    )

                    if nested_processed:
                        matched_items.append(nested_processed)
                        self.logger.debug(
                            "Nested processing succeeded. Matched item: %s",
                            nested_processed,
                        )
                    else:
                        self.logger.debug(
                            "Nested processing did not produce any matches for item: %s",
                            item,
                        )
                    unmatched_items.update(nested_unmatched)
                    errors.extend(nested_errors)
                else:
                    self.logger.debug(
                        "Item is not a dictionary. Attempting to coerce type. Item: %s, Expected Type: %s",
                        item,
                        expected_type,
                    )
                    coerced_item = self._coerce_item_type(item, expected_type)

                    if coerced_item is not None and isinstance(
                        coerced_item, expected_type
                    ):
                        matched_items.append(coerced_item)
                        self.logger.debug(
                            "Matched list item '%s' at path '%s' to type '%s'.",
                            item,
                            item_path,
                            expected_type,
                        )
                    else:
                        self.logger.warning(
                            "Failed to coerce item '%s' at path '%s'. Expected type: '%s'.",
                            item,
                            item_path,
                            expected_type,
                        )
                        errors.append({item_path: item})
            except (ValueError, TypeError) as e:
                self.logger.error(
                    "Failed to coerce item '%s' at path '%s' to type '%s': Error: %s",
                    item,
                    item_path,
                    expected_type,
                    str(e),
                )
                errors.append({item_path: item})

        # Log a warning if multiple entities are detected in the list
        if len(matched_items) > 1:
            self.logger.warning(
                "Detected multiple entities at path '%s'. Consider updating the schema to include a parent key for the nested structure.",
                current_path,
            )

        self.logger.debug(
            "Returning matched_items: %s, unmatched_items: %s, and errors: %s",
            matched_items,
            unmatched_items,
            errors,
        )

        return matched_items, unmatched_items, errors

    def _validate_list_item(
        self, item, item_schema, matched_items, unmatched_items, item_path
    ):
        expected_type = self.schema_handler.get_type_from_field(item_schema)
        coerced_item = self._coerce_item_type(item, item_schema)

        if isinstance(coerced_item, expected_type):
            matched_items.append(coerced_item)
            self.logger.debug(
                "Matched list item '%s' at path '%s' to type '%s'.",
                item,
                item_path,
                expected_type,
            )
        else:
            unmatched_items[item_path] = [item]

    def _coerce_item_type(self, item, expected_type):
        if not expected_type:
            self.logger.warning(
                "Expected type for item '%s' is undefined. Returning item as is.", item
            )
            return item

        try:
            if expected_type == bool:
                if isinstance(item, str) and item.lower() in ["true", "false"]:
                    coerced_item = item.lower() == "true"
                    self.logger.debug(
                        "Coercion - item: %s, expected type: %s, coerced to: %s",
                        item,
                        expected_type,
                        coerced_item,
                    )
                    return coerced_item
                elif isinstance(item, int):
                    coerced_item = bool(item)
                    self.logger.debug(
                        "Coercion - integer to boolean: item: %s, coerced to: %s",
                        item,
                        coerced_item,
                    )
                    return coerced_item
            elif expected_type == str:
                coerced_item = str(item)
                self.logger.debug(
                    "Coercion - item: %s, expected type: %s, coerced to: %s",
                    item,
                    expected_type,
                    coerced_item,
                )
                return coerced_item
            elif expected_type in ["number", float, int]:
                self.logger.debug(
                            "Running coercion on: '%s' expected type: %s",
                            item,
                            expected_type,
                        ) # TODO: Delete after debugging
                if isinstance(item, str):
                      
                    # Try standard numeric conversion
                    try:
                        coerced_item = float(item) if "." in item else int(item)
                        self.logger.debug(
                            "Standard conversion succeeded for '%s': %s",
                            item,
                            coerced_item,
                        )
                        return coerced_item
                    except ValueError:
                        self.logger.debug(
                            "Standard conversion failed for '%s'. Attempting word2number.",
                            item,
                        )

                    # Try text-to-number conversion
                    try:
                        coerced_item = w2n.word_to_num(item)
                        self.logger.debug(
                            "word2number conversion succeeded for '%s': %s",
                            item,
                            coerced_item,
                        )
                        return float(coerced_item) if expected_type == float else int(coerced_item)
                    except ValueError:
                        self.logger.warning(
                            "word2number conversion failed for '%s'. Returning original value.",
                            item,
                        )
                        return item

                if expected_type == float:
                    # Convert explicitly to float
                    coerced_item = float(item)
                    self.logger.debug(
                        "Coerced item '%s' to float: %s", item, coerced_item
                    )
                    return coerced_item

                if expected_type == int:
                    # Convert explicitly to int
                    coerced_item = (
                        int(round(item)) if isinstance(item, float) else int(item)
                    )
                    self.logger.debug(
                        "Coerced item '%s' to int: %s", item, coerced_item
                    )
                    return coerced_item
            elif isinstance(item, expected_type):
                self.logger.debug(
                    "Coercion for item: %s unnecessary. Item is expected type %s",
                    item,
                    expected_type,
                )
                return item
            else:
                self.logger.debug(
                    "Coercion for item: %s of expected type %s is not implemented.",
                    item,
                    expected_type,
                )
                return expected_type(item)
        except (ValueError, TypeError):
            self.logger.error(
                "Failed to coerce item '%s' to type '%s'.", item, expected_type
            )

        self.logger.debug("No coercion applied for item: %s. Returning as is.", item)
        return item
