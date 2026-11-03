import numpy as np

np.set_printoptions(precision=3, suppress=True)


# ==========================================
# STEP 1: CREATE TOKEN REPRESENTATIONS
# ==========================================

tokens = ["The", "cat", "sat", "down"]

X = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0]
])

print("Input representations:")
print(X)

print("\nInput shape:", X.shape)


# ==========================================
# STEP 2: CREATE QUERY, KEY, VALUE MATRICES
# ==========================================

# Simplified, manually chosen projection matrices.
# Real Transformer models learn these weights.

W_Q = np.eye(3)
W_K = np.eye(3)
W_V = np.eye(3)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("\nQuery shape:", Q.shape)
print("Key shape:", K.shape)
print("Value shape:", V.shape)


# ==========================================
# STEP 3: CALCULATE ATTENTION SCORES
# ==========================================

d_k = K.shape[1]

scores = Q @ K.T / np.sqrt(d_k)

print("\nAttention scores:")
print(scores)


# ==========================================
# STEP 4: APPLY SOFTMAX TO EACH ROW
# ==========================================

def softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)

    exp_values = np.exp(shifted)

    return exp_values / np.sum(
        exp_values,
        axis=-1,
        keepdims=True
    )


attention_weights = softmax(scores)

print("\nAttention weights:")
print(attention_weights)


# ==========================================
# STEP 5: COMBINE THE VALUE VECTORS
# ==========================================

output = attention_weights @ V

print("\nSelf-attention output:")
print(output)

print("\nOutput shape:", output.shape)


# ==========================================
# STEP 6: DISPLAY WHO ATTENDS TO WHOM
# ==========================================

print("\nAttention map:")

print("Each row is the querying token.")
print("Each column is the token receiving attention.\n")

print("             ", end="")

for token in tokens:
    print(f"{token:>9}", end="")

print()

for token, row in zip(tokens, attention_weights):
    print(f"{token:>12}", end="")

    for weight in row:
        print(f"{weight:9.3f}", end="")

    print()