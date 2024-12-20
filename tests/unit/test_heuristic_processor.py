import pytest
from openai_json.heuristic_processor import HeuristicProcessor
from openai_json.schema_handler import SchemaHandler


@pytest.fixture
def schema_handler():
    handler = SchemaHandler()
    return handler


@pytest.fixture
def heuristic_processor(schema_handler):
    return HeuristicProcessor(schema_handler)


def test_heuristic_processor_valid(heuristic_processor, schema_handler):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
    }
    schema_handler.submit_schema(schema)

    data = {"name": "John", "age": 30, "email": "john@example.com"}
    processed, unmatched = heuristic_processor.process(data)

    assert processed == data
    assert unmatched == []


def test_heuristic_processor_unmatched_keys(heuristic_processor, schema_handler):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    schema_handler.submit_schema(schema)

    data = {"name": "John Doe", "age": "thirty"}  # Invalid type for age
    processed, unmatched = heuristic_processor.process(data)

    assert processed == {"name": "John Doe"}
    assert unmatched == ["age"]


def test_heuristic_processor_nested_schema(heuristic_processor, schema_handler):
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "details": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer"},
                            "email": {"type": "string"},
                        },
                        "required": ["age"],
                    },
                },
                "required": ["name", "details"],
            },
        },
    }
    schema_handler.submit_schema(schema)

    data = {
        "user": {
            "name": "John Doe",
            "details": {"age": 30, "email": "john@example.com"},
        }
    }
    processed, unmatched = heuristic_processor.process(data)

    expected_processed = {
        "user": {
            "name": "John Doe",
            "details": {"age": 30, "email": "john@example.com"},
        }
    }
    assert processed == expected_processed
    assert unmatched == []


def test_heuristic_processor_list_handling():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }
    data = {"items": ["item1", 42, "item3"]}  # 42 does not match the "string" type
    heuristic_processor = HeuristicProcessor(schema_handler=SchemaHandler())
    heuristic_processor.schema_handler.submit_schema(schema)

    processed, unmatched = heuristic_processor.process(data)
    assert processed == {"items": ["item1", "item3"]}
    assert unmatched == [{"items[1]": [42]}]


def test_heuristic_processor_string_to_list(heuristic_processor, schema_handler):
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }
    schema_handler.submit_schema(schema)

    data = {"tags": "tag1, tag2, tag3"}
    processed, unmatched = heuristic_processor.process(data)

    assert processed == {"tags": ["tag1", "tag2", "tag3"]}
    assert unmatched == []


def test_heuristic_processor_normalized_keys(heuristic_processor, schema_handler):
    schema = {
        "type": "object",
        "properties": {
            "Key_A": {"type": "string"},
            "Key_B": {"type": "integer"},
        },
    }
    schema_handler.submit_schema(schema)

    data = {"key a": "value1", "KEY B": 100}
    processed, unmatched = heuristic_processor.process(data)

    assert processed == {"key_a": "value1", "key_b": 100}
    assert unmatched == []
