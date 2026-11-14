import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load API key
load_dotenv(Path(__file__).parent / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)


# -------------------------
# TOOLS
# -------------------------

def calculate(expression: str) -> str:
    """Calculate a basic arithmetic expression."""

    import ast
    import operator

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv
    }

    def evaluate(node):

        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):
            operation = operators.get(type(node.op))

            if operation is None:
                raise ValueError("Unsupported operator")

            return operation(
                evaluate(node.left),
                evaluate(node.right)
            )

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return evaluate(node.operand)

        raise ValueError("Invalid expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree.body))

    except Exception as error:
        return f"Calculation error: {error}"


def get_current_date() -> str:
    """Return today's current date."""

    from datetime import datetime

    return datetime.now().strftime("%d %B %Y")


# -------------------------
# AGENT
# -------------------------

tools = [calculate, get_current_date]

config = types.GenerateContentConfig(
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=False,
        maximum_remote_calls=5
    ),
    system_instruction=(
        "You are a helpful AI assistant. "
        "Use the calculator for arithmetic. "
        "Use the date tool for current-date questions. "
        "Never invent tool results."
    )
)


print("\nAI Agent Ready!")
print("Type 'exit' to quit.")


while True:

    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        continue

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=question,
            config=config
        )

        print("\nAgent:")
        print(response.text or "No text response returned.")

    except Exception as error:
        print("\nError:", error)