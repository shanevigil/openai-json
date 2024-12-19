from openai_json.openai_json import OpenAI_JSON

# Replace with your actual OpenAI API key
API_KEY = "your-api-key"

client = OpenAI_JSON(API_KEY)

# Define the schema
schema = {"name": str, "age": int, "email": str}

# Query to send
query = "Generate a JSON object with a person's name, age, and email."

# Process the request
response = OpenAI_JSON.handle_request(query, schema)
print(response)
