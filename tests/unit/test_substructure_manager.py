import pytest
from openai_json.substructure_manager import SubstructureManager


def test_store_and_retrieve_unmatched_keys(schema_handler):
    manager = SubstructureManager(schema_handler)
    data = {"key_one": "value1", "key_two": "value2"}  # Match normalized form
    unmatched_keys = ["key_one", "key_two"]

    manager.store_unmatched_keys(unmatched_keys, data)
    unmatched_data = manager.retrieve_unmatched_data()

    assert unmatched_data == {
        "key_one": "value1",
        "key_two": "value2",
    }


def test_store_and_retrieve_unmatched_keys_nested_structure(mock_schema_handler):
    manager = SubstructureManager(mock_schema_handler)
    data = {
        "level1": {
            "level2": {"key2": "value2"},
        },
        "key3": "value3",
    }
    unmatched_keys = ["level1.level2.key2", "key3"]

    manager.store_unmatched_keys(unmatched_keys, data)
    unmatched_data = manager.retrieve_unmatched_data()

    assert unmatched_data == {
        "level1.level2.key2": "value2",
        "key3": "value3",
    }
