from openai_json.output_assembler import OutputAssembler
from openai_json.schema_handler import SchemaHandler
import pytest


def test_assemble_output_with_unmatched_data(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {"name": "John", "age": 30}
    unmatched_data = {"extra_key": "extra_value"}

    final_output = assembler.assemble_output(processed_data, unmatched_data)

    assert final_output["processed_data"] == processed_data
    assert final_output["unmatched_data"] == unmatched_data


def test_assemble_output_without_unmatched_data(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {"name": "John", "age": 30}
    unmatched_data = {}

    final_output = assembler.assemble_output(processed_data, unmatched_data)

    assert final_output["processed_data"] == processed_data
    assert "unmatched_data" not in final_output


def test_empty_processed_and_unmatched_data(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {}
    unmatched_data = {}

    final_output = assembler.assemble_output(processed_data, unmatched_data)

    assert final_output["processed_data"] == processed_data
    assert "unmatched_data" not in final_output


def test_assemble_output_with_nested_unmatched_data(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {"name": "John"}
    unmatched_data = {"extra_key": {"nested_key": "nested_value"}}

    final_output = assembler.assemble_output(processed_data, unmatched_data)

    assert final_output["processed_data"] == processed_data
    assert final_output["unmatched_data"] == unmatched_data


def test_assemble_output_preserves_original_keys(mock_schema_handler):
    assembler = OutputAssembler(schema_handler=mock_schema_handler)
    processed_data = {"user_name": "Alice", "user_age": 25}
    unmatched_data = {"extra_key": "extra_value"}

    final_output = assembler.assemble_output(processed_data, unmatched_data)

    assert final_output["processed_data"] == processed_data
    assert final_output["unmatched_data"] == unmatched_data
