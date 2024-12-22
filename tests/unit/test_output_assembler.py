from openai_json.output_assembler import OutputAssembler
from openai_json.schema_handler import SchemaHandler
import pytest


def test_assemble_output_with_all_fields(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {"name": "John", "age": 30}
    unmatched_data = [{"extra_key": "extra_value"}]
    transformed_data = {"extra_key": "transformed_value"}
    errors = [{"error_key": "error_value"}]

    final_output = assembler.assemble_output(
        processed_data, transformed_data, unmatched_data, errors
    )

    assert final_output["processed_data"] == {
        "name": "John",
        "age": 30,
        "extra_key": "transformed_value",
    }
    assert final_output["unmatched_data"] == unmatched_data
    assert final_output["error"] == errors


def test_assemble_output_with_errors_only(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {}
    unmatched_data = {}
    transformed_data = {}
    errors = [{"error_key": "error_value"}]

    final_output = assembler.assemble_output(
        processed_data, transformed_data, unmatched_data, errors
    )

    assert final_output["processed_data"] == {}
    assert final_output["unmatched_data"] == []
    assert final_output["error"] == errors


def test_assemble_output_with_transformed_and_unmatched_data(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {}
    unmatched_data = [{"extra_key": "extra_value"}]
    transformed_data = {"extra_key": "transformed_value"}
    errors = []

    final_output = assembler.assemble_output(
        processed_data, transformed_data, unmatched_data, errors
    )

    # Transformed data should take precedence
    assert final_output["processed_data"] == {"extra_key": "transformed_value"}
    assert final_output["unmatched_data"] == unmatched_data
    assert final_output["error"] == errors


def test_assemble_output_with_empty_inputs(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {}
    unmatched_data = {}
    transformed_data = {}
    errors = []

    final_output = assembler.assemble_output(
        processed_data, transformed_data, unmatched_data, errors
    )

    assert final_output["processed_data"] == {}
    assert final_output["unmatched_data"] == []
    assert final_output["error"] == []


def test_assemble_output_with_nested_data(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {"user": {"name": "Alice"}}
    unmatched_data = [{"extra_key": {"nested_key": "nested_value"}}]
    transformed_data = {}
    errors = []

    final_output = assembler.assemble_output(
        processed_data, transformed_data, unmatched_data, errors
    )

    assert final_output["processed_data"] == processed_data
    assert final_output["unmatched_data"] == unmatched_data
    assert final_output["error"] == []


def test_assemble_output_handles_key_mapping(mock_schema_handler):
    # Mock schema handler's map_keys_to_original method
    mock_schema_handler.submit_schema(
        {f"Key {k}": f"value{v}" for k, v in enumerate(range(1, 5), start=1)}
    )

    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {"key_1": "value1"}
    unmatched_data = [{"key_2": "value2"}]
    transformed_data = {"key_3": "value3"}
    errors = [{"key_4": "error_value"}]

    final_output = assembler.assemble_output(
        processed_data, transformed_data, unmatched_data, errors
    )

    # Verify all keys are mapped back to their original forms
    assert final_output["processed_data"] == {
        "Key 1": "value1",
        "Key 3": "value3",
    }
    assert final_output["unmatched_data"] == [{"Key 2": "value2"}]
    assert final_output["error"] == [{"Key 4": "error_value"}]
