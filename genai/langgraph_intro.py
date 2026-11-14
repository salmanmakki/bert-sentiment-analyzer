from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define shared state
class AgentState(TypedDict):
    question: str
    response: str


# 2. Create first node
def understand_question(state: AgentState):

    question = state["question"]

    return {
        "response": f"I received your question: {question}"
    }


# 3. Create second node
def generate_answer(state: AgentState):

    response = state["response"]

    return {
        "response": response + "\nProcessing completed!"
    }


# 4. Create graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "understand",
    understand_question
)

graph_builder.add_node(
    "answer",
    generate_answer
)


# 5. Connect nodes
graph_builder.add_edge(START, "understand")

graph_builder.add_edge("understand", "answer")

graph_builder.add_edge("answer", END)


# 6. Compile graph
agent = graph_builder.compile()


# 7. Run graph
result = agent.invoke({
    "question": "What is machine learning?",
    "response": ""
})


print(result)