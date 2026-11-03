import numpy as np


# ==========================================
# STEP 1: REPRESENT THREE WORDS AS VECTORS
# ==========================================

words = ["amazing", "acting", "terrible"]

word_vectors = np.array([
    [1.0, 0.0, 0.0],   # amazing
    [0.0, 1.0, 0.0],   # acting
    [0.0, 0.0, 1.0]    # terrible
])

print("Word vectors:")
print(word_vectors)

print("\nShape:", word_vectors.shape)


# ==========================================
# STEP 2: CREATE A QUERY
# ==========================================

query = np.array([0.0, 0.0, 1.0])

print("\nQuery:")
print(query)


# ==========================================
# STEP 3: CALCULATE ATTENTION SCORES
# ==========================================

scores = word_vectors @ query

print("\nAttention scores:")
print(scores)


# ==========================================
# STEP 4: CONVERT SCORES INTO WEIGHTS
# ==========================================

def softmax(values):
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values)


attention_weights = softmax(scores)

print("\nAttention weights:")

for word, weight in zip(words, attention_weights):
    print(f"{word:10s}: {weight:.4f}")

print("\nSum of attention weights:")
print(np.sum(attention_weights))


# ==========================================
# STEP 5: COMBINE THE WORD VECTORS
# ==========================================

context_vector = attention_weights @ word_vectors

print("\nContext vector:")
print(context_vector)