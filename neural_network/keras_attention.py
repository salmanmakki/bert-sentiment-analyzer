import tensorflow as tf
from tensorflow.keras.layers import Input, MultiHeadAttention

# ==========================================
# STEP 1: CREATE ARTIFICIAL TOKEN INPUTS
# ==========================================

# 1 sentence, 4 tokens, 8 features per token
X = tf.random.normal(shape=(1, 4, 8))

print("Input shape:", X.shape)


# ==========================================
# STEP 2: CREATE MULTI-HEAD ATTENTION
# ==========================================

attention_layer = MultiHeadAttention(
    num_heads=2,
    key_dim=4
)


# ==========================================
# STEP 3: APPLY SELF-ATTENTION
# ==========================================

output, attention_scores = attention_layer(
    query=X,
    key=X,
    value=X,
    return_attention_scores=True
)


# ==========================================
# STEP 4: INSPECT THE RESULTS
# ==========================================

print("\nOutput shape:", output.shape)

print("Attention scores shape:", attention_scores.shape)

print("\nAttention scores:")
print(attention_scores.numpy())


# ==========================================
# STEP 5: INSPECT THE TRAINABLE PARAMETERS
# ==========================================

print("\nTrainable parameters:")

for variable in attention_layer.trainable_variables:
    print(variable.name, variable.shape)

print(
    "\nTotal trainable parameters:",
    attention_layer.count_params()
)