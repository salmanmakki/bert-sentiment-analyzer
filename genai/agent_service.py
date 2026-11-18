import os
from datetime import datetime
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing.")

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        api_version="v1beta"
    )
)


# TOOL 1: Calculator
def calculator(a, b, operation):
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "Cannot divide by zero."
        return a / b

    return "Unsupported operation."


# TOOL 2: Current date and time
def get_datetime():
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


# TOOL REGISTRY
available_functions = {
    "calculator": calculator,
    "get_datetime": get_datetime
}


# TOOL DECLARATIONS
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="calculator",
                description="Perform basic mathematical calculations.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "a": types.Schema(type=types.Type.NUMBER),
                        "b": types.Schema(type=types.Type.NUMBER),
                        "operation": types.Schema(
                            type=types.Type.STRING,
                            enum=["add", "subtract", "multiply", "divide"]
                        )
                    },
                    required=["a", "b", "operation"]
                )
            ),
            types.FunctionDeclaration(
                name="get_datetime",
                description="Get the current date, time and day."
            )
        ]
    )
]


# AGENT LOOP
def run_agent(question):

    conversation = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
        )
    ]

    max_iterations = 5

    for _ in range(max_iterations):

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=conversation,
            config=types.GenerateContentConfig(
                tools=tools
            )
        )

        conversation.append(response.candidates[0].content)

        function_calls = response.function_calls

        if not function_calls:
            return response.text

        tool_responses = []

        for function_call in function_calls:

            function = available_functions.get(function_call.name)

            if function is None:
                result = "Unknown tool."
            else:
                try:
                    result = function(**function_call.args)
                except Exception as error:
                    result = f"Tool execution failed: {error}"

            tool_responses.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={"result": result}
                )
            )

        conversation.append(
            types.Content(
                role="user",
                parts=tool_responses
            )
        )

    return "Maximum tool iterations reached."