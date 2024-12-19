from openai_json.substructure_manager import SubstructureManager


def test_store_and_retrieve_unmatched_keys():
    manager = SubstructureManager()
    data = {"key1": "value1", "key2": "value2", "key3": "value3"}
    unmatched_keys = ["key2", "key3"]

    manager.store_unmatched_keys(unmatched_keys, data)
    unmatched_data = manager.retrieve_unmatched_data()

    assert unmatched_data == {"key2": "value2", "key3": "value3"}


def test_clear_unmatched_data():
    manager = SubstructureManager()
    data = {"key1": "value1", "key2": "value2"}
    unmatched_keys = ["key2"]

    manager.store_unmatched_keys(unmatched_keys, data)
    assert manager.retrieve_unmatched_data() == {"key2": "value2"}

    manager.clear()
    assert manager.retrieve_unmatched_data() == {}
