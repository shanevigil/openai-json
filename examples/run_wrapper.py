from openai_json.wrapper import Wrapper

# Replace with your actual OpenAI API key
API_KEY = "your-api-key"

wrapper = Wrapper(API_KEY)

# Define the schema
schema = {"name": str, "age": int, "email": str}

# Query to send
query = "Generate a JSON object with a person's name, age, and email."

# Process the request
response = wrapper.handle_request(query, schema)
print(response)
