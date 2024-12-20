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
4. [Development](#development)
5. [Contributing](#contributing)
6. [License](#license)

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

### Example Script

Here's an example script to get started:

```python
from openai_json.openai_json import OpenAI_JSON

# Initialize the OpenAI_JSON
api_key = "your-openai-api-key"
schema = {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "email": {"type": "string"}
}

client = OpenAI_JSON(gpt_api_key=api_key, schema=schema)

# Send a query
query = "Provide information about a user including their name, age, and email."
response = client.handle_request(query)

# Print the structured output
print(response)
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

## Development

### Running Tests

To ensure everything works as expected, run the test suite using `pytest`:

```bash
pytest tests/
```

### Training the ML Model

For handling complex JSON variations, you can train a custom machine learning model. Place the model in the `data/ml_model/` directory. An example training script is provided in `examples/`.

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

