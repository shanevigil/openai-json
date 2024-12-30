# tests/conftest.py
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from openai_json.schema_handler import SchemaHandler
import logging


def pytest_configure(config):
    # Configure the root logger to display DEBUG level messages
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
    )


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Fixture to mock both sync and async OpenAI clients and their response behavior."""
    # Create separate mock clients for sync and async
    sync_mock_client = MagicMock()
    async_mock_client = AsyncMock()

    # Function to set mock responses dynamically for both sync and async clients
    def set_mock_response(mock_content):
        # Mock response for sync client
        sync_response = MagicMock()
        sync_response.choices = [MagicMock(message=MagicMock(content=mock_content))]
        sync_mock_client.chat.completions.create.return_value = sync_response

        # Mock response for async client
        async_response = AsyncMock()
        async_response.choices = [MagicMock(message=MagicMock(content=mock_content))]
        async_mock_client.chat.completions.create.return_value = async_response

    # Base system message; updated in tests as needed
    expected_system_message_base = "Respond in valid JSON format."

    # Patch both sync and async clients in the target module
    monkeypatch.setattr(
        "openai_json.api_interface.OpenAI", lambda api_key: sync_mock_client
    )
    monkeypatch.setattr(
        "openai_json.api_interface.AsyncOpenAI", lambda api_key: async_mock_client
    )

    return (
        sync_mock_client,
        async_mock_client,
        set_mock_response,
        expected_system_message_base,
    )


@pytest.fixture
def schema_handler():
    """Fixture to provide a fresh instance of SchemaHandler."""
    return SchemaHandler()


@pytest.fixture
def mock_schema_handler():
    """
    Provides a mock SchemaHandler object for testing purposes.
    """
    schema_handler = SchemaHandler()
    schema_handler.submit_schema(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
        }
    )

    # Generate example JSON for this schema
    example_json = schema_handler.generate_example_json()

    return schema_handler, example_json
