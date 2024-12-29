import pytest
from openai_json.heuristic_processor import HeuristicProcessor
import json


@pytest.fixture
def heuristic_processor(schema_handler):
    return HeuristicProcessor(schema_handler)


# Parameterized tests with ID specified for easier reference
test_cases = [
    {
        "id": "Simple Test 1: valid response",
        "schema": {
            "Key A": {"type": "integer"},
            "Key B": {"type": "string"},
            "Key C": {"type": "list"},
        },
        "response": '{"Key A": 231, "Key B": "Mary has a dog.", "Key C": ["Max"]}',
        "expected": {
            "processed_data": {
                "key_a": 231,
                "key_b": "Mary has a dog.",
                "key_c": ["Max"],
            },
            "unmatched_data": {},
            "error": {},
        },
    },
    {
        "id": "Simple Test 2: unmatched key response",
        "schema": {
            "Key A": {"type": "integer"},
            "Key B": {"type": "string"},
        },
        "response": '{"Key A": 231, "Key B": "Mary has a dog.", "Key C": ["Max"]}',
        "expected": {
            "processed_data": {
                "key_a": 231,
                "key_b": "Mary has a dog.",
            },
            "unmatched_data": {
                "key_c": [
                    "Max",
                ]
            },
            "error": {},
        },
    },
    {
        "id": "Simple Test 3: Normalized key processing",
        "schema": {
            "Key With Spaces": {"type": "string"},
            "ALL CAPS": {"type": "string"},
            "keysWithCamelCase": {"type": "string"},
            "keys_with_underscore": {"type": "string"},
        },
        "response": '{"Key With Spaces": "Some Value", "ALL CAPS": "Some Value", "keysWithCamelCase": "Some Value", "keys_with_underscore": "Some Value"}',
        "expected": {
            "processed_data": {
                "key_with_spaces": "Some Value",
                "all_caps": "Some Value",
                "keys_with_camel_case": "Some Value",
                "keys_with_underscore": "Some Value",
            },
            "unmatched_data": {},
            "error": {},
        },
    },
    {
        "id": "List Test 1: Valid response with list items",
        "schema": {
            "Key A": {"type": "integer"},
            "Key B": {"type": "string"},
            "Key C": {"type": "list", "items": {"type": "string"}},
        },
        "response": '{"Key A": 231, "Key B": "Mary has three dogs.", "Key C": ["Max","Spot","Rover"]}',
        "expected": {
            "processed_data": {
                "key_a": 231,
                "key_b": "Mary has three dogs.",
                "key_c": ["Max", "Spot", "Rover"],
            },
            "unmatched_data": {},
            "error": {},
        },
    },
    {
        "id": "Coercion Test 1: Coercion of list in string format to list",
        "schema": {
            "Key A": {"type": "list"},
        },
        "response": '{"Key A": "tag1, tag2, tag3"}',
        "expected": {
            "processed_data": {
                "key_a": ["tag1", "tag2", "tag3"],
            },
            "unmatched_data": {},
            "error": {},
        },
    },
    {
        "id": "Coercion Test 2: float, int, str, and boolean",
        "schema": {
            "Key A": {"type": "integer"},
            "Key B": {"type": "integer"},
            "Key C": {"type": "number"},
            "Key D": {"type": "number"},
            "Key E": {"type": "string"},
            "Key F": {"type": "string"},
            "Key G": {"type": "boolean"},
            "Key H": {"type": "boolean"},
            "Key I": {"type": "boolean"},
            "Key J": {"type": "boolean"},
        },
        "response": (
            '{"Key A": "231", "Key B": 231.3, "Key C": "231.4", "Key D": 231.3, '
            '"Key E": 42, "Key F": 42.2, "Key G": "true", "Key H": "false", '
            '"Key I": 1, "Key J": 0}'
        ),
        "expected": {
            "processed_data": {
                "key_a": 231,
                "key_b": 231,
                "key_c": 231.4,
                "key_d": 231.3,
                "key_e": "42",
                "key_f": "42.2",
                "key_g": True,
                "key_h": False,
                "key_i": True,
                "key_j": False,
            },
            "unmatched_data": {},
            "error": {},
        },
    },
    {
        "id": "Coercion Test 2: Mixed data within lists",
        "schema": {
            "Key A": {"type": "list", "items": {"type": "string"}},
            "Key B": {"type": "list", "items": {"type": "integer"}},
            "Key C": {
                "type": "list",
            },  # No type specified
        },
        "response": '{"Key A": ["two", 2], "Key B": [1, "2", 3, "four"], "Key C": [1, 2, 3, "4", "five"]}',
        "expected": {
            "processed_data": {
                "key_a": ["two", "2"],
                "key_b": [1, 2, 3, 4],
                "key_c": [1, 2, 3, 4, 5],
            },
            "unmatched_data": {},
            "error": {},
        },
    },
    {
        "id": "Coercion Test 3: Invalid string-to-integer generates error",
        "schema": {
            "key_a": {"type": "integer"},
        },
        "response": '{"key_a": "abc"}',
        "expected": {
            "processed_data": {},
            "unmatched_data": {},
            "error": {"key_a": "abc"},
        },
    },
    {
        "id": "Nesting Test 1: Nested data in list",
        "schema": {
            "Key A": {"type": "integer"},
            "Key B": {"type": "string"},
            "Key C": {"type": "list", "items": {"type": "string"}},
        },
        "response": '{"Key A": 231, "Key B": "Mary has some dogs.", "Dogs": {"Key C": ["Max", "Rover", 3]}}',
        "expected": {
            "processed_data": {
                "key_a": 231,
                "key_b": "Mary has some dogs.",
                "key_c": ["Max", "Rover", "3"],
            },
            "unmatched_data": {},  # Dogs parent is erased as the child key was found
            "error": {},
        },
    },
    {
        "id": "Nesting Test 2: Complex nested object schema",
        "schema": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string"},
        },
        "response": '{"nested": {'
        '    "person": {'
        '        "name": "Johnathan",'
        '        "properties": {'
        '            "alias": "Johnny",'
        '            "details": {'
        '                "age": 30,'
        '                "contact info": {'
        '                    "phone": "888-888-8888",'
        '                    "email": "john@example.com"'
        "                }"
        "            }"
        "        }"
        "    }"
        "}}",
        "expected": {
            "processed_data": {
                "name": "Johnathan",
                "age": 30,
                "email": "john@example.com",
            },
            "unmatched_data": {
                "nested.person.properties.alias": "Johnny",
                "nested.person.properties.details.contact_info.phone": "888-888-8888",
            },
            "error": {},
        },
    },
    {
        "id": "Coercion Test 4: Word-to-number conversion fallback",
        "schema": {
            "Key A": {"type": "number"},
        },
        "response": '{"Key A": "forty-two"}',
        "expected": {
            "processed_data": {"key_a": 42},
            "unmatched_data": {},
            "error": {},
        },
    },
    # TODO: The following test fails because w2n just extracts the numeric word and ignores the rest. Research and implement a way to handle appropriate edges. Exclusion (any number but) makes sense, but do others? About 50 should be 50. If the field is numeric, no qualifiers are possible anyway, (e.g. "< 50", or "45>x>50", etc.). There is probably no json spec for "range" or inequality.
    # {
    #     "id": "Coercion Test N: Unsupported text returns original value",
    #     "schema": {
    #         "Key A": {"type": "number"},
    #     },
    #     "response": '{"Key A": "any number but fifty"}',
    #     "expected": {
    #         "processed_data": {},
    #         "unmatched_data": {},
    #         "error": {"key_a": "any number but fifty"},
    #     },
    # },
    {
        "id": "Coercion Test 5: Invalid string for number",
        "schema": {
            "Key A": {"type": "number"},
        },
        "response": '{"Key A": "not a number"}',
        "expected": {
            "processed_data": {},
            "unmatched_data": {},
            "error": {"key_a": "not a number"},
        },
    },
    {
        "id": "Coercion Test 6: Mixed standard and word-to-number",
        "schema": {
            "Key A": {"type": "list", "items": {"type": "integer"}},
            "Key B": {"type": "list", "items": {"type": "number"}},
        },
        "response": '{"Key A": ["42", "forty-two", "not a number"], "Key B": [42, "Forty Two"]}',
        "expected": {
            "processed_data": {
                "key_a": [42, 42],  # Expected coerced integers
                "key_b": [42.0, 42.0],  # Expected floats
            },
            "unmatched_data": {},  # No unmatched data
            "error": {
                "key_a[2]": "not a number",  # `not a number` should remain an error
            },
        },
    },
    # TODO: IF the schema specifies a one to many relationship, nesting should work, e.g. people to person. However, if no nested structure is specified and a one to many relationship is returned, return the first record, and all everything else to unmatched, and log a warning.
    # {
    #     "id": "Nesting Test 3: Handles multiple entities despite single-schema field",
    #     "schema": {"name": {"type": "string"}},
    #     "response": '{"people": [{"person": {"name": "Alice"}}, {"person": {"name": "Bob"}}]}',
    #     "expected": {
    #         "processed_data": {"name": "Alice"},
    #         "unmatched_data": {{"name": "Bob"}},
    #         "error": {},
    #     },
    # },
]


# Dynamically parameterize tests
@pytest.mark.parametrize(
    "schema, chatgpt_response, expected_output",
    [(case["schema"], case["response"], case["expected"]) for case in test_cases],
    ids=[case["id"] for case in test_cases],
)
def test_heuristic_processor_integration(
    schema_handler, heuristic_processor, schema, chatgpt_response, expected_output
):
    """
    Integration test for HeuristicProcessor with various schemas and ChatGPT responses.
    """
    # Mock schema validation (if required)
    schema_handler.submit_schema(schema)

    # Convert the mocked ChatGPT response to a Python dictionary
    parsed_response = json.loads(chatgpt_response)

    # Process the parsed response using HeuristicProcessor
    result = heuristic_processor.process(parsed_response)

    # Assert the processed data matches the expected output
    assert result.matched == expected_output["processed_data"]

    # Assert unmatched_keys matches expected unmatched_data
    assert result.unmatched == expected_output["unmatched_data"]

    # Assert errors matches expected error
    assert result.errors == expected_output["error"]
