from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)


# Actual Python function
def multiply(a: float, b: float) -> float:
    return a * b


# Declare function for Gemini
multiply_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="multiply",
            description="Multiply two numbers.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "a": {"type": "NUMBER"},
                    "b": {"type": "NUMBER"}
                },
                "required": ["a", "b"]
            }
        )
    ]
)

# First request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is 25 multiplied by 40?",
    config=types.GenerateContentConfig(
        tools=[multiply_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )
)

# Inspect function call
function_call = response.function_calls[0]

print("Function requested:", function_call.name)
print("Arguments:", function_call.args)

# Execute Python function
if function_call.name == "multiply":

    result = multiply(
        function_call.args["a"],
        function_call.args["b"]
    )

    print("Function result:", result)

    # Send result back to Gemini
    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="What is 25 multiplied by 40?"
                    )
                ]
            ),
            response.candidates[0].content,
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="multiply",
                        response={"result": result}
                    )
                ]
            )
        ],
        config=types.GenerateContentConfig(
            tools=[multiply_tool]
        )
    )

    print("\nAI:", final_response.text)