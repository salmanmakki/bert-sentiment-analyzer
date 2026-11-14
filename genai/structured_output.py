from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)

# Define expected output
config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema={
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING"},
            "education": {"type": "STRING"},
            "skills": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "experience_level": {"type": "STRING"}
        },
        "required": [
            "name",
            "education",
            "skills",
            "experience_level"
        ]
    }
)

# Generate structured response
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="""
    Extract information from this text:

    Salman is a BTech CSE student.
    He knows Python, JavaScript and React.
    He is a beginner in Generative AI.
    """,
    config=config
)

# Convert JSON string into Python dictionary
data = json.loads(response.text)

print(json.dumps(data, indent=4))