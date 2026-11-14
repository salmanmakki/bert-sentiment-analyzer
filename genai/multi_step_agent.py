import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# API configuration
load_dotenv(Path(__file__).parent / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)


# TOOL 1: Calculate total price
def calculate_total(price: float, quantity: int) -> float:
    """Calculate total price before discount."""
    return price * quantity


# TOOL 2: Calculate discount
def calculate_discount(total: float, percentage: float) -> float:
    """Calculate discount amount."""
    return total * percentage / 100


# TOOL 3: Calculate final price
def calculate_final_price(total: float, discount: float) -> float:
    """Calculate final price after discount."""
    return total - discount


# Register tools
tools = [
    calculate_total,
    calculate_discount,
    calculate_final_price
]


config = types.GenerateContentConfig(
    tools=tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=False,
        maximum_remote_calls=10
    ),
    system_instruction="""
    You are a shopping assistant.

    When calculating a discounted price:
    1. Calculate the total using calculate_total.
    2. Calculate the discount using calculate_discount.
    3. Calculate the final price using calculate_final_price.

    Always use the tools for calculations.
    Explain the results clearly.
    """
)


# Agent loop
print("Multi-Step AI Agent Ready!")
print("Type exit to stop.")


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
        print(response.text or "No response generated.")

    except Exception as error:
        print("\nError:", error)