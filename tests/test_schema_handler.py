import pytest
from openai_json.schema_handler import SchemaHandler


def test_submit_valid_schema():
    handler = SchemaHandler()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    handler.submit_schema(schema)
    assert handler.schema == schema


def test_submit_invalid_schema():
    handler = SchemaHandler()
    # Invalid schema with unsupported type
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
    result, message = handler.validate_data(data)
    assert result is True
    assert message == "Validation passed."


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
