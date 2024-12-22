# OpenAI-JSON: A Python Library for Structured JSON Responses

![Build Status](https://github.com/shanevigil/openai-json/actions/workflows/test-build.yml/badge.svg)

OpenAI-JSON is a Python package designed to streamline interactions with OpenAI's ChatGPT API by ensuring JSON responses conform to user-defined schemas. The package employs a hybrid approach using heuristic rules and machine learning to process JSON structures, handle variations, and deliver structured outputs.

## Key Features

- **Schema Management**: Submit and validate schemas for JSON responses.
- **API Integration**: Seamlessly interact with OpenAI's ChatGPT API.
- **Heuristic Processing**: Quickly align straightforward JSON structures to the schema.
- **Machine Learning Assistance**: Handle complex JSON variations using a pre-trained ML model.
- **Comprehensive Logging**: Track unmatched keys and variations for debugging.
- **Modular Design**: Easily extend or integrate with other tools and workflows.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Directory Structure](#directory-structure)
4. [Validation and Error Handling](#validation-and-error-handling)
5. [Development](#development)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/openai-json.git
cd openai-json
```

### Set Up the Python Environment

#### Using Conda

```bash
conda env create --file environment.yml
conda activate ./env
```

#### Using pip

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic Workflow

1. **Define a Schema**: Create a schema that defines the expected JSON structure.
2. **Send a Query**: Use the `OpenAI_JSON` class to send a query to OpenAI's ChatGPT API.
3. **Receive Structured Output**: The `OpenAI_JSON` processes the response to conform to the schema.

### Examples

#### Instantiating OpenAI-JSON with a Schema

You can directly provide a schema when instantiating the `OpenAI_JSON` object. This simplifies the initialization process.

```python
from openai_json.openai_json import OpenAI_JSON

api_key = "your-api-key"
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"}
    },
    "required": ["name", "email"]
}

client = OpenAI_JSON(gpt_api_key=api_key, schema=schema)
response = client.handle_request("Provide information about a user.")

print("Structured Output:", response)
print("Unmatched Data:", client.unmatched_data)  # Data not aligning with the schema
print("Errors:", client.errors)  # Coercion failures or unexpected issues
```

#### Generating Outputs with Prompts

You can add field-specific prompts to your schema using `extract_prompts`. For instance:

```python
schema_with_prompts = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "prompt": "Please provide the user's full name."},
        "email": {"type": "string", "prompt": "Enter a valid email address."}
    }
}

handler = SchemaHandler(schema_with_prompts)
print(handler.extract_prompts())
```

#### Handling API Queries

```python
from openai_json.openai_json import OpenAI_JSON

api_key = "your-api-key"
schema = {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "email": {"type": "string"}
}

client = OpenAI_JSON(gpt_api_key=api_key, schema=schema)
response = client.handle_request("Provide information about a user.")
print("Structured Output:", response)
```

#### Modifying Default Prefix for Prompts

The default prefix for schema prompts can be customized:

```python
handler = SchemaHandler(schema_with_prompts)
print(handler.extract_prompts(prefix="User data collection:")
```

---

## Directory Structure

The project follows a modular and organized structure:

```bash
openai_json/
├── openai_json/               # Core package
├── tests/                     # Unit and integration tests
├── data/                      # Schemas, ML models, and examples
├── docs/                      # Documentation
├── examples/                  # Example scripts
├── env/                       # Conda environment
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment configuration
├── README.md                  # Project overview and usage
└── setup.py                   # Package installation script
```

---

## Validation and Error Handling

The OpenAI-JSON library includes robust mechanisms to handle validation errors, unmatched data, and coercion-related issues. These features ensure that JSON responses conform to the schema while providing detailed insights into mismatches and processing errors.

### Validation Errors

Validation errors occur when the schema or response from ChatGPT fails to comply with valid JSON. These will be logged.

### Unmatched Data and Coercion Errors

When using the `OpenAI_JSON` object, any unmatched data or errors in coercion are automatically stored in the instance attributes `unmatched_data` and `errors`. These represent responses from ChatGPT that are too different from the schema to be managed programmatically. This allows for detailed review and debugging.

Example:

```python
from openai_json.openai_json import OpenAI_JSON

api_key = "your-api-key"
schema = {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "email": {"type": "string"}
}

client = OpenAI_JSON(gpt_api_key=api_key, schema=schema)
response = client.handle_request("Provide information about a user.")

print("Structured Output:", response)
print("Unmatched Data:", client.unmatched_data)  # Data not aligning with the schema
print("Errors:", client.errors)  # Coercion failures or unexpected issues
```

---

## Development

### Training the Machine Learning Model

The `MachineLearningProcessor` supports schema-compliant transformations using a pre-trained model. Currently, the model is not trained. This is the next major step before version 1.0 is officially released. Once trained, the model should be saved in `data/ml_model/` and loaded into the processor as demonstrated in `ml_processor.py`:

```python
from openai_json.ml_processor import MachineLearningProcessor

processor = MachineLearningProcessor("data/ml_model/model.pkl")
unmatched_keys = {"unknown_field": "example_value"}
transformed = processor.predict_transformations(unmatched_keys)
print("Transformed:", transformed)
```

### Running Tests

To ensure everything works as expected, run the test suite using `pytest`:

```bash
pytest tests/
```

---

## Contributing

We welcome contributions to improve the package! To contribute:

1. Fork the repository and create a feature branch.
2. Make your changes and write tests.
3. Submit a pull request.

Please review the contributing guidelines for detailed instructions.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or support, please open an issue on [GitHub](https://github.com/yourusername/openai-json).

