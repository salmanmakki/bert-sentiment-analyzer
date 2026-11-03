import numpy as np

np.set_printoptions(precision=3, suppress=True)

# ==========================================
# STEP 1: CREATE FOUR TOKEN REPRESENTATIONS
# ==========================================

tokens = ["The", "cat", "sat", "down"]

X = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

print("Input shape:", X.shape)


# ==========================================
# STEP 2: CREATE TWO ATTENTION HEADS
# ==========================================

# These are fixed matrices for demonstration.
# Real Transformers learn these weights.

W_Q1 = np.eye(4)[:, :2]
W_K1 = np.eye(4)[:, :2]
W_V1 = np.eye(4)[:, :2]

W_Q2 = np.eye(4)[:, 2:]
W_K2 = np.eye(4)[:, 2:]
W_V2 = np.eye(4)[:, 2:]


# ==========================================
# STEP 3: DEFINE SCALED DOT-PRODUCT ATTENTION
# ==========================================

def attention(Q, K, V):

    scores = Q @ K.T / np.sqrt(K.shape[-1])

    scores = scores - np.max(
        scores,
        axis=-1,
        keepdims=True
    )

    exp_scores = np.exp(scores)

    weights = exp_scores / np.sum(
        exp_scores,
        axis=-1,
        keepdims=True
    )

    output = weights @ V

    return output, weights


# ==========================================
# STEP 4: RUN THE FIRST ATTENTION HEAD
# ==========================================

Q1 = X @ W_Q1
K1 = X @ W_K1
V1 = X @ W_V1

head1_output, head1_weights = attention(Q1, K1, V1)

print("\nHead 1 output shape:", head1_output.shape)

print("\nHead 1 attention weights:")
print(head1_weights)


# ==========================================
# STEP 5: RUN THE SECOND ATTENTION HEAD
# ==========================================

Q2 = X @ W_Q2
K2 = X @ W_K2
V2 = X @ W_V2

head2_output, head2_weights = attention(Q2, K2, V2)

print("\nHead 2 output shape:", head2_output.shape)

print("\nHead 2 attention weights:")
print(head2_weights)


# ==========================================
# STEP 6: COMBINE THE HEADS
# ==========================================

combined_output = np.concatenate(
    [head1_output, head2_output],
    axis=-1
)

print("\nCombined output:")
print(combined_output)

print("\nCombined output shape:", combined_output.shape)