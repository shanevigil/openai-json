from openai_json.ml_processor import MachineLearningProcessor
import pytest

# from unittest.mock import patch, MagicMock


def test_predict_misspellings(schema_handler):
    processor = MachineLearningProcessor(schema_handler)

    unmatched_data = {
        "user_emial": "test@example.com",
        "usr_email": "user@example.com",
        "contactnumbr": "123-456-7890",
    }
    schema_keys = ["user_email", "contact_number", "name"]

    expected_output = {
        "user_email": "test@example.com",
        "user_email": "user@example.com",
        "contact_number": "123-456-7890",
    }

    output = processor._predict_misspellings(unmatched_data, schema_keys)
    assert output == expected_output


def test_predict_synonyms(schema_handler):
    processor = MachineLearningProcessor(schema_handler)

    unmatched_data = {
        "email_address": "user@example.com",
        "phone_number": "123-456-7890",
    }
    schema_keys = ["user_email", "contact_number"]

    expected_output = {
        "user_email": "user@example.com",
        "contact_number": "123-456-7890",
    }

    output = processor._predict_synonyms(unmatched_data, schema_keys)
    assert output == expected_output


def test_predict_contextual_matching():
    processor = MachineLearningProcessor(schema_handler=None)

    unmatched_data = {
        "profile_summary": "An enthusiastic coder.",
    }

    schema = {
        "account_id": {
            "type": "integer",
            "prompt": "Provide a unique identifier for a user, for example: 12345",
        },
        "email": {
            "type": "string",
            "prompt": "Provide the user's email address.",
        },
        "full_name": {
            "type": "string",
            "prompt": "Provide a default username.",
        },
        "profile_description": {
            "type": "string",
            "prompt": "Provide a brief biography of the user's life.",
        },
    }

    expected_output = {
        "profile_description": "An enthusiastic coder.",
    }

    output = processor._predict_contextual_matching(unmatched_data, schema)
    assert output == expected_output


# def test_predict_contextual_matching_with_rich_schema(schema_handler):
#     processor = MachineLearningProcessor(schema_handler)

#     unmatched_data = {
#         "random_key": "value",
#     }

#     schema_keys = [
#         {
#             "name": "user_id",
#             "type": "identifier",
#             "prompt": "Unique identifier for a user",
#             "example": "12345",
#         },
#         {
#             "name": "profile_name",
#             "type": "string",
#             "prompt": "Name displayed on user profile",
#             "example": "johndoe",
#         },
#         {
#             "name": "email",
#             "type": "string",
#             "prompt": "User's email address",
#             "example": "example@example.com",
#         },
#     ]

#     expected_output = {}  # No transformed data if no match is found

#     output = processor._predict_contextual_matching(unmatched_data, schema_keys)
#     assert output == expected_output


def test_process_item_method(schema_handler):

    schema = {
        "account_id": {
            "type": "integer",
            "prompt": "Provide a unique identifier for a user, for example: 12345",
        },
        "email": {
            "type": "string",
            "prompt": "Provide the user's email address.",
        },  # Expected Fuzzy Match
        "profile name": {
            "type": "string",
            "prompt": "Provide a default username.",
        },  # Expected Synonym Match
        "profile description": {
            "type": "string",
            "prompt": "Provide a description of the user.",
        },  # Expected contextual Match
    }

    schema_handler.submit_schema(schema)

    processor = MachineLearningProcessor(schema_handler)

    unmatched_data = {
        "emial": "test@example.com",
        "person name": "John Doe",
        "biography": "John is a person of unknown origins.",
    }

    expected_output = {
        "email": "test@example.com",
        "profile name": "John Doe",
        "profile description": "John is a person of unknown origins.",
    }

    processed_data, unmatched_keys, errors = processor.process(unmatched_data)
    assert processed_data == expected_output
    assert (
        unmatched_keys == unmatched_data
    )  # This should not change as it adjusted later in the assembler
    assert not errors
