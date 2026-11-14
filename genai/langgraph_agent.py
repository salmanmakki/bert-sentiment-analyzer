import os
import operator

from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# --------------------------------
# GEMINI CONFIGURATION
# --------------------------------

load_dotenv(Path(__file__).parent / ".env")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1beta"}
)


# --------------------------------
# DEFINE TOOLS
# --------------------------------

def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression."""

    allowed_operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv
    }

    # Safe arithmetic parser
    import ast

    def evaluate(node):

        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):

            operation = allowed_operators.get(
                {
                    ast.Add: "+",
                    ast.Sub: "-",
                    ast.Mult: "*",
                    ast.Div: "/"
                }.get(type(node.op))
            )

            if operation is None:
                raise ValueError("Unsupported operation")

            return operation(
                evaluate(node.left),
                evaluate(node.right)
            )

        raise ValueError("Invalid expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree.body))

    except Exception as error:
        return f"Calculation error: {error}"


tools = [calculator]

gemini_tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="calculator",
                description="Calculate basic arithmetic expressions.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "expression": types.Schema(
                            type="STRING",
                            description="Arithmetic expression, e.g. 245 * 678"
                        )
                    },
                    required=["expression"]
                )
            )
        ]
    )
]


# --------------------------------
# LANGGRAPH STATE
# --------------------------------

class AgentState(TypedDict):

    messages: Annotated[list, operator.add]


# --------------------------------
# GEMINI NODE
# --------------------------------

def agent_node(state: AgentState):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=state["messages"],
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=(
                "You are a helpful AI assistant. "
                "Use the calculator tool for arithmetic questions. "
                "If no tool is needed, answer directly."
            )
        )
    )

    return {
        "messages": [response.candidates[0].content]
    }


# --------------------------------
# BUILD GRAPH
# --------------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge("tools", "agent")

graph = builder.compile()


# --------------------------------
# CHAT LOOP
# --------------------------------

print("\nLangGraph AI Agent Ready!")
print("Type 'exit' to stop.")

while True:

    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit"]:
        break

    if not question:
        continue

    try:

        result = graph.invoke({
            "messages": [
                types.Content(
                    role="user",
                    parts=[types.Part(text=question)]
                )
            ]
        })

        last_message = result["messages"][-1]

        answer = "\n".join(
            part.text
            for part in last_message.parts or []
            if part.text
        )

        print("\nAgent:", answer)

    except Exception as error:

        print("\nError:", error)