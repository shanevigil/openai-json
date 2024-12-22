import pytest
from openai_json.substructure_manager import SubstructureManager


def test_store_and_retrieve_unmatched_data(schema_handler):
    manager = SubstructureManager(schema_handler)
    unmatched_data = [
        {"key_one": "value1"},
        {"key_two": "value2"},
    ]  # Already normalized

    manager.store_unmatched_data(unmatched_data)
    retrieved_normalized_data = manager.retrieve_unmatched_data()

    assert retrieved_normalized_data == [{"key_one": "value1"}, {"key_two": "value2"}]


def test_store_and_retrieve_unmatched_data_nested_structure(schema_handler):
    manager = SubstructureManager(schema_handler)
    unmatched_data = [
        {"level1.level2.key2": "value2"},
        {"key3": "value3"},
    ]

    manager.store_unmatched_data(unmatched_data)
    retrieved_normalized_data = manager.retrieve_unmatched_data()

    assert retrieved_normalized_data == [
        {"level1.level2.key2": "value2"},
        {"key3": "value3"},
    ]


def test_store_and_retrieve_error_data(schema_handler):
    manager = SubstructureManager(schema_handler)
    errors = [
        {"key1": "invalid_value"},
        {"key2": "another_invalid_value"},
    ]

    manager.store_error_data(errors)
    retrieved_errors = manager.retrieve_error_data()

    assert retrieved_errors == [
        {"key1": "invalid_value"},
        {"key2": "another_invalid_value"},
    ]


def test_reconcile_transformed_data(schema_handler):
    manager = SubstructureManager(schema_handler)

    # Set up unmatched data store
    manager.unmatched_data_store = [
        {"key1": "value1"},
        {"key2": "value2"},
        {"key3": "value3"},
    ]

    transformed_data = [
        {
            "key1": "new_value1",
        },
        {
            "key4": "new_value4",
        },
    ]

    # Reconcile transformed data
    manager.reconcile_transformed_data(transformed_data)

    # Assert unmatched_data_store is updated
    assert manager.unmatched_data_store == [
        {"key2": "value2"},
        {"key3": "value3"},
    ]


def test_reconcile_transformed_data_with_list_structure(schema_handler):
    manager = SubstructureManager(schema_handler)

    # Set up unmatched data store as a list of dictionaries
    manager.unmatched_data_store = [
        {"key1": "value1"},
        {"key2": "value2"},
        {"key3": "value3"},
    ]

    transformed_data = {
        "key1": "new_value1",
        "key3": "new_value3",
    }

    # Reconcile transformed data
    manager.reconcile_transformed_data(transformed_data)

    # Assert unmatched_data_store no longer contains reconciled keys
    assert manager.unmatched_data_store == [
        {"key2": "value2"},
    ]
