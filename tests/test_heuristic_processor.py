from openai_json.heuristic_processor import HeuristicProcessor
from openai_json.schema_handler import SchemaHandler


def test_heuristic_processor_valid():
    schema_handler = SchemaHandler()
    schema = {"name": str, "age": int, "email": str}
    schema_handler.submit_schema(schema)
    processor = HeuristicProcessor(schema_handler)

    data = {"name": "John", "age": 30, "email": "john@example.com"}
    processed, unmatched = processor.process(data)
    assert processed == data
    assert unmatched == []


def test_heuristic_processor_unmatched_keys():
    schema_handler = SchemaHandler()
    schema = {"name": str, "age": int}
    schema_handler.submit_schema(schema)
    processor = HeuristicProcessor(schema_handler)

    # Data contains an extra key (email) and is missing a required key (age)
    data = {"name": "John", "email": "john@example.com"}
    processed, unmatched = processor.process(data)

    assert processed == {"name": "John"}
    assert unmatched == ["email", "age"]
