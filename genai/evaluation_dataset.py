import json

data = [
    {
        "question": "What is supervised learning?",
        "reference": "Supervised learning trains a model using labelled training data."
    },
    {
        "question": "What is lexical analysis?",
        "reference": "Lexical analysis is the first phase of a compiler. It converts source code into tokens."
    },
    {
        "question": "What is syntax analysis?",
        "reference": "Syntax analysis checks the grammatical structure of tokens according to programming language grammar."
    },
    {
        "question": "What is a neural network?",
        "reference": "A neural network is a machine learning model consisting of interconnected artificial neurons."
    },
    {
        "question": "What is code optimization?",
        "reference": "Code optimization improves code efficiency without changing its intended functionality."
    }
]

with open("evaluation_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Evaluation dataset created!")
print("Total questions:", len(data))