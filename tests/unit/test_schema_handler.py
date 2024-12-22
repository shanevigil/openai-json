import pytest
from openai_json.schema_handler import SchemaHandler
from datetime import datetime


def test_submit_valid_schema():
    handler = SchemaHandler()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    handler.submit_schema(schema)
    expected_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    assert handler.normalized_schema == expected_schema


def test_schema_handler_submit_json_string():
    handler = SchemaHandler()
    json_string_schema = '{"Key A": "integer", "Key B": "string"}'
    handler.submit_schema(json_string_schema)
    expected_schema = {
        "key_a": {"type": "integer"},  # Normalized keys
        "key_b": {"type": "string"},
    }
    assert handler.normalized_schema == expected_schema


def test_schema_handler_invalid_format():
    handler = SchemaHandler()
    invalid_format = ["not", "a", "valid", "schema"]  # A list instead of a dict

    with pytest.raises(
        ValueError, match="Unsupported schema format: Expected a dictionary, got: list"
    ):
        handler.submit_schema(invalid_format)


def test_submit_invalid_schema():
    handler = SchemaHandler()
    invalid_schema = {
        "type": "object",
        "properties": {"name": {"type": "unsupported_type"}},  # Invalid type
    }
    with pytest.raises(
        ValueError,
        match="'unsupported_type' is not valid under any of the given schemas",
    ):
        handler.submit_schema(invalid_schema)


def test_validate_data_valid():
    handler = SchemaHandler()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    handler.submit_schema(schema)
    data = {"name": "John Doe", "age": 30}
    result, processed_data = handler.validate_data(data)
    assert result is True
    assert (
        processed_data == data
    )  # Expecting the returned processed data to match input


def test_validate_data_invalid():
    handler = SchemaHandler()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    handler.submit_schema(schema)
    invalid_data = {"name": "John Doe", "age": "thirty"}  # Incorrect type for age
    result, message = handler.validate_data(invalid_data)
    assert result is False
    assert "Validation failed" in message


def test_schema_handler_submit_simplified_schema():
    handler = SchemaHandler()
    simplified_schema = {
        "Key A": "integer",
        "Key B": "string",
        "Key C": "list",
    }
    handler.submit_schema(simplified_schema)
    expected_schema = {
        "key_a": {"type": "integer"},  # Normalized keys
        "key_b": {"type": "string"},
        "key_c": {"type": "list"},
    }
    assert handler.normalized_schema == expected_schema


def test_schema_handler_submit_detailed_schema():
    handler = SchemaHandler()
    detailed_schema = {
        "Key A": {"type": "integer"},
        "Key B": {"type": "string"},
        "Key C": {"type": "list"},
    }
    handler.submit_schema(detailed_schema)
    expected_schema = {
        "key_a": {"type": "integer"},  # Normalized keys
        "key_b": {"type": "string"},
        "key_c": {"type": "list"},
    }
    assert handler.normalized_schema == expected_schema


def test_schema_handler_submit_python_types():
    handler = SchemaHandler()
    schema_with_python_types = {
        "Key A": int,
        "Key B": str,
        "Key C": list,
    }
    handler.submit_schema(schema_with_python_types)
    expected_schema = {
        "key_a": {"type": "integer"},  # Normalized keys
        "key_b": {"type": "string"},
        "key_c": {"type": "list"},
    }
    assert handler.normalized_schema == expected_schema


def test_register_type_valid():
    handler = SchemaHandler()

    # Register a new type
    handler.register_type(datetime, "string")

    # Check that the mapping is updated
    assert (
        handler.python_type_mapping[datetime] == "string"
    ), "The type-to-JSON mapping failed."

    # Check reverse mapping
    reverse_mapping = {v: k for k, v in handler.python_type_mapping.items()}
    assert (
        reverse_mapping["string"] == datetime
    ), "The JSON-to-type reverse mapping failed."

    # Ensure existing mappings are not affected
    assert (
        handler.python_type_mapping[str] == "string"
    ), "The 'str' mapping was altered."
    assert (
        handler.python_type_mapping[int] == "integer"
    ), "The 'int' mapping was altered."

    # Test re-registration behavior (optional, depends on the `register_type` method implementation)
    handler.register_type(datetime, "timestamp")
    assert (
        handler.python_type_mapping[datetime] == "timestamp"
    ), "Re-registration of the type failed."
    reverse_mapping = {v: k for k, v in handler.python_type_mapping.items()}
    assert (
        reverse_mapping["timestamp"] == datetime
    ), "The reverse mapping after re-registration failed."


def test_register_type_invalid():
    handler = SchemaHandler()
    with pytest.raises(
        ValueError, match="Invalid type mapping. Expected \\(type, str\\)."
    ):
        handler.register_type("not_a_type", "string")
    with pytest.raises(
        ValueError, match="Invalid type mapping. Expected \\(type, str\\)."
    ):
        handler.register_type(datetime, 123)  # Invalid JSON type


def test_add_field_to_empty_schema():
    handler = SchemaHandler()
    handler.submit_schema({"type": "object", "properties": {}})
    handler.add_field("new_field", {"type": "integer"})
    assert "new_field" in handler.normalized_schema["properties"]
    assert handler.normalized_schema["properties"]["new_field"] == {"type": "integer"}


def test_add_field_with_invalid_field_name():
    handler = SchemaHandler()
    handler.submit_schema({"type": "object", "properties": {}})
    with pytest.raises(
        ValueError, match="Invalid field name or schema. Expected \\(str, dict\\)."
    ):
        handler.add_field(123, {"type": "integer"})


def test_add_field_with_invalid_field_schema():
    handler = SchemaHandler()
    handler.submit_schema({"type": "object", "properties": {}})
    with pytest.raises(
        ValueError, match="Invalid field name or schema. Expected \\(str, dict\\)."
    ):
        handler.add_field("new_field", "not_a_dict")


def test_diff_schema_added_field():
    handler = SchemaHandler()
    handler.submit_schema(
        {"type": "object", "properties": {"field1": {"type": "string"}}}
    )
    new_schema = {
        "type": "object",
        "properties": {"field1": {"type": "string"}, "field2": {"type": "integer"}},
    }
    diff = handler.diff_schema(new_schema)
    assert diff["added"] == {"field2": {"type": "integer"}}
    assert diff["removed"] == {}
    assert diff["changed"] == {}


def test_diff_schema_removed_field():
    handler = SchemaHandler()
    handler.submit_schema(
        {
            "type": "object",
            "properties": {"field1": {"type": "string"}, "field2": {"type": "integer"}},
        }
    )
    new_schema = {"type": "object", "properties": {"field1": {"type": "string"}}}
    diff = handler.diff_schema(new_schema)
    assert diff["added"] == {}
    assert diff["removed"] == {"field2": {"type": "integer"}}
    assert diff["changed"] == {}


def test_diff_schema_changed_field():
    handler = SchemaHandler()
    handler.submit_schema(
        {"type": "object", "properties": {"field1": {"type": "string"}}}
    )
    new_schema = {"type": "object", "properties": {"field1": {"type": "integer"}}}
    diff = handler.diff_schema(new_schema)
    assert diff["added"] == {}
    assert diff["removed"] == {}
    assert diff["changed"] == {"field1": ({"type": "string"}, {"type": "integer"})}


def test_diff_schema_no_changes():
    handler = SchemaHandler()
    handler.submit_schema(
        {"type": "object", "properties": {"field1": {"type": "string"}}}
    )
    new_schema = {"type": "object", "properties": {"field1": {"type": "string"}}}
    diff = handler.diff_schema(new_schema)
    assert diff["added"] == {}
    assert diff["removed"] == {}
    assert diff["changed"] == {}


def test_diff_schema_invalid_format():
    handler = SchemaHandler()
    handler.submit_schema(
        {"type": "object", "properties": {"field1": {"type": "string"}}}
    )
    with pytest.raises(
        ValueError, match="Invalid schema format. Expected a dictionary."
    ):
        handler.diff_schema("not_a_dict")


def test_diff_schema_no_existing_schema():
    handler = SchemaHandler()
    new_schema = {"type": "object", "properties": {"field1": {"type": "string"}}}
    with pytest.raises(
        ValueError, match="No schema has been submitted yet. Cannot compare schemas."
    ):
        handler.diff_schema(new_schema)


def test_extract_prompts():
    """
    Unit test for SchemaHandler's extract_prompts method.
    Ensures prompts are correctly identified and formatted.
    """
    handler = SchemaHandler()

    # Submit schema with prompts
    schema_with_prompts = {
        "Key A": {"type": "string", "prompt": "Provide the user's full name."},
        "Key B": {"type": "integer", "prompt": "Provide the user's age."},
        "Key C": {"type": "boolean"},  # No prompt
    }
    handler.submit_schema(schema_with_prompts)

    # Extract prompts
    prompts = handler.extract_prompts()

    # Expected formatted string
    expected_prompts = (
        "Here are the field-specific instructions:\n"
        "Key A: Provide the user's full name.\n"
        "Key B: Provide the user's age."
    )

    assert prompts == expected_prompts, f"Unexpected prompts: {prompts}"


def test_extract_prompts_no_prompts():
    """
    Test extract_prompts when the schema has no prompts.
    """
    handler = SchemaHandler()

    # Submit schema without prompts
    schema_without_prompts = {
        "Key A": {"type": "string"},
        "Key B": {"type": "integer"},
    }
    handler.submit_schema(schema_without_prompts)

    # Extract prompts
    prompts = handler.extract_prompts()

    # Expected result is an empty string
    assert prompts == "", f"Expected no prompts but got: {prompts}"


def test_extract_prompts_with_custom_prefix():
    """
    Test extract_prompts with a custom prefix.
    """
    handler = SchemaHandler()

    schema_with_prompts = {
        "Key A": {"type": "string", "prompt": "Provide the user's full name."},
        "Key B": {"type": "integer", "prompt": "Provide the user's age."},
    }
    handler.submit_schema(schema_with_prompts)

    custom_prefix = "Custom instructions for the fields:"
    prompts = handler.extract_prompts(prefix=custom_prefix)

    expected_prompts = (
        "Custom instructions for the fields:\n"
        "Key A: Provide the user's full name.\n"
        "Key B: Provide the user's age."
    )
    assert prompts == expected_prompts, f"Unexpected prompts: {prompts}"


def test_extract_prompts_no_prompts_no_prefix():
    """
    Test extract_prompts when no prompts exist in the schema.
    Ensure the prefix is not included in the output.
    """
    handler = SchemaHandler()

    schema_without_prompts = {
        "Key A": {"type": "string"},
        "Key B": {"type": "integer"},
    }
    handler.submit_schema(schema_without_prompts)

    prompts = handler.extract_prompts()

    assert prompts == "", f"Expected no prompts but got: {prompts}"


def test_extract_prompts_no_prompts_with_custom_prefix():
    """
    Test extract_prompts with no prompts and a custom prefix.
    Ensure the prefix is not included in the output.
    """
    handler = SchemaHandler()

    schema_without_prompts = {
        "Key A": {"type": "string"},
        "Key B": {"type": "integer"},
    }
    handler.submit_schema(schema_without_prompts)

    custom_prefix = "Custom prefix for testing:"
    prompts = handler.extract_prompts(prefix=custom_prefix)

    assert prompts == "", f"Expected no prompts but got: {prompts}"
