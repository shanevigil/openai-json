import pytest
from openai_json.data_manager import DataManager, ResultData
from unittest.mock import Mock

class TestDataManager:
    def test_add_result_updates_state(self, mock_schema_handler):
        # Initialize DataManager with mock schema handler
        data_manager = DataManager(mock_schema_handler)

        # Create a mock ResultData
        result = ResultData(
            matched={"key1": "value1"},
            unmatched={"key2": "value2"},
            errors={"key3": "value3"},
        )

        # Add the result to the DataManager
        data_manager.add_result(result)

        # Assert matched, unmatched, and errors are updated correctly
        assert data_manager.matched == {"key1": "value1"}
        assert data_manager.unmatched == {"key2": "value2"}
        assert data_manager.errors == {"key3": "value3"}

    def test_add_multiple_results_updates_correctly(self, mock_schema_handler):
        # Initialize DataManager with mock schema handler
        data_manager = DataManager(mock_schema_handler)

        # Add first result
        result1 = ResultData(
            matched={"key1": "value1"},
            unmatched={"key2": "value2"},
            errors={"key3": "value3"},
        )
        data_manager.add_result(result1)

        # Add second result with overlapping keys
        result2 = ResultData(
            matched={"key2": "new_value2","key3": "value3"},
            unmatched={"key4": "value4"},
            errors={"key5": "value5"},
        )
        data_manager.add_result(result2)

        # Assert final state
        assert data_manager.matched == {"key1": "value1", "key2": "new_value2","key3": "value3"}
        assert data_manager.unmatched == {"key4": "value4"}
        assert data_manager.errors == {"key5": "value5"}

    def test_finalize_output(self, mock_schema_handler):
        # Initialize DataManager with mock schema handler
        data_manager = DataManager(mock_schema_handler)

        # Add some matched data
        result = ResultData(
            matched={"key1": "value1", "key2": "value2"},
            unmatched={},
            errors={},
        )
        data_manager.add_result(result)

        # Finalize output
        output = data_manager.finalize_output()

        # Assert that the output maps normalized keys to original keys
        expected_output = {
            "original_key1": "value1",
            "original_key2": "value2",
        }
        assert output == expected_output

    def test_reconcile(self, mock_schema_handler):
        # Initialize DataManager with mock schema handler
        data_manager = DataManager(mock_schema_handler)

        # Add conflicting results
        result1 = ResultData(
            matched={"key1": "value1"},
            unmatched={"key2": "value2"},
            errors={"key3": "value3"},
        )
        result2 = ResultData(
            matched={"key1": "conflicting_value"},
            unmatched={"key4": "value4"},
            errors={"key5": "value5"},
        )
        data_manager.add_result(result1)
        data_manager.add_result(result2)

        # Call reconcile
        data_manager._reconcile()

        # Assert reconciled state
        assert data_manager.matched == {"key1": "conflicting_value"}
        assert data_manager.unmatched == {"key4": "value4"}
        assert data_manager.errors == {"key5": "value5"}

    def test_empty_state_after_initialization(self, mock_schema_handler):
        # Initialize DataManager with mock schema handler
        data_manager = DataManager(mock_schema_handler)

        # Assert initial state is empty
        assert data_manager.matched == {}
        assert data_manager.unmatched == {}
        assert data_manager.errors == {}
        assert data_manager.results == []

    def test_logging_during_updates(self, mock_schema_handler):
        # Initialize DataManager with mock schema handler
        data_manager = DataManager(mock_schema_handler)

        # Mock logger
        data_manager.logger = Mock()

        # Add a result
        result = ResultData(
            matched={"key1": "value1"},
            unmatched={"key2": "value2"},
            errors={"key3": "value3"},
        )
        data_manager.add_result(result)

        # Assert logger was called
        data_manager.logger.debug.assert_called()

