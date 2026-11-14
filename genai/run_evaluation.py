import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from google import genai

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

# Load API key
load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY missing from .env")

# Gemini client
client = genai.Client(api_key=api_key)

# RAGAS evaluation model
llm = llm_factory(
    "gemini-2.5-flash",
    provider="google",
    client=client
)

# Test dataset
data = {
    "question": [
        "What is supervised learning?",
        "What is lexical analysis?"
    ],

    "answer": [
        "Supervised learning trains a model using labelled training data.",
        "Lexical analysis is the first phase of a compiler that converts source code into tokens."
    ],

    "contexts": [
        [
            "Supervised learning is a machine learning approach in which models learn from labelled examples."
        ],
        [
            "Lexical analysis is the first phase of a compiler. It reads source code and generates tokens."
        ]
    ],

    "ground_truth": [
        "Supervised learning trains a model using labelled training data.",
        "Lexical analysis converts source code into tokens."
    ]
}

dataset = Dataset.from_dict(data)

# Evaluation metrics
metrics = [
    Faithfulness(llm=llm),
    AnswerRelevancy(llm=llm),
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm)
]

print("\nStarting RAG evaluation...\n")

results = evaluate(
    dataset=dataset,
    metrics=metrics
)

print("\nEvaluation Results:")
print(results)

results.to_pandas().to_csv(
    "rag_evaluation_results.csv",
    index=False
)

print("\nResults saved to rag_evaluation_results.csv")