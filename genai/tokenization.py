from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

# Input
text = """
Generative AI is changing software development.
I am learning how Large Language Models work.
"""

# Tokenize
tokens = tokenizer.tokenize(text)

# Convert to IDs
token_ids = tokenizer.encode(text)

# Decode
decoded_text = tokenizer.decode(token_ids)

print("Original Text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Total Tokens:", len(token_ids))
print("Decoded Text:", decoded_text)