# tests/conftest.py
from unittest.mock import patch, MagicMock
import pytest


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client and its response."""
    with patch("openai.OpenAI") as mock_openai:
        # Create a mock client instance
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Provide a way to set the mock response dynamically
        def set_mock_response(mock_content):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content=mock_content))]
            mock_client.chat.completions.create.return_value = mock_response

        # Track the system message
        expected_system_message = "Respond in valid JSON format."

        # Wrap the client and additional data
        yield mock_client, set_mock_response, expected_system_message
