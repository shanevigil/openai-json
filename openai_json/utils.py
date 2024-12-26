from openai_json.schema_handler import SchemaHandler


def build_path(path: str, key: str) -> str:
    """
    Constructs a hierarchical path string by appending a key to an existing path.

    Args:
        path (str): The existing hierarchical path (e.g., "parent.child").
        key (str): The key to append to the path.

    Returns:
        str: The resulting hierarchical path. If the path is empty, returns the key itself.

    Example:
        >>> build_path("parent.child", "key")
        'parent.child.key'
        >>> build_path("", "key")
        'key'
    """
    return f"{path}.{key}" if path else key


def get_field_definition(schema: dict, normalized_key: str):
    """
    Retrieves the field definition for a given normalized key from a schema.

    Args:
        schema (dict): The schema containing field definitions.
        normalized_key (str): The normalized key whose definition is to be retrieved.

    Returns:
        dict or None: The field definition for the normalized key, or None if not found.

    Example:
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        >>> get_field_definition(schema, "name")
        {'type': 'string'}
        >>> get_field_definition(schema, "unknown_key")
        None
    """
    return schema.get("properties", {}).get(normalized_key, schema.get(normalized_key))


def normalize_response_data(data: dict) -> dict:
    """
    Normalizes the keys of a response data dictionary by applying a normalization function.

    Args:
        data (dict): The original data dictionary with raw keys.

    Returns:
        dict: A new dictionary with keys normalized using `SchemaHandler.normalize_text`.

    Example:
        data = {"User Name": "John Doe", "AGE": 30}
        SchemaHandler.normalize_text = lambda x: x.lower().replace(" ", "_")
        >>> normalize_response_data(data)
        {'user_name': 'John Doe', 'age': 30}
    """
    return {SchemaHandler.normalize_text(key): value for key, value in data.items()}


def get_key_from_path(path: str) -> str:
    """
    Extracts the key from a given path.

    Args:
        path (str): The hierarchical path string.

    Returns:
        str: The key extracted from the path.
    """
    if "." in path:
        return path.split(".")[-1]
    return path


def build_nested_dict(path: str, value) -> dict:
    """
    Creates a nested dictionary structure based on the given path.

    Args:
        path (str): The hierarchical path string (e.g., "parent.child.key").
        value: The value to set at the deepest key in the nested dictionary.

    Returns:
        dict: A nested dictionary representing the path.
    """
    keys = path.split(".")
    nested_dict = current = {}
    for key in keys[:-1]:
        current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return nested_dict


def add_nested_path(existing_dict: dict, path: str, value) -> None:
    """
    Adds a nested path with a value to an existing dictionary.

    Args:
        existing_dict (dict): The dictionary to update.
        path (str): The hierarchical path string (e.g., "parent.child.key").
        value: The value to set at the deepest key in the path.

    Returns:
        None: The dictionary is updated in-place.
    """
    keys = path.split(".")
    current = existing_dict
    for key in keys[:-1]:
        # Navigate the path, adding empty dictionaries if necessary
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    # Set the value at the deepest key
    current[keys[-1]] = value
