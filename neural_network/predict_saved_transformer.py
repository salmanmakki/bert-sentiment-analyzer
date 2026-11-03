import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. MODEL SETTINGS
# ==========================================

VOCAB_SIZE = 10_000
MAX_LENGTH = 200
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR.parent / "transformer_sentiment.weights.h5"


# ==========================================
# 2. TOKEN + POSITIONAL EMBEDDINGS
# ==========================================

class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, vocab_size, max_length, embed_dim):
        super().__init__()

        self.token_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=embed_dim
        )

    def call(self, token_ids):

        sequence_length = tf.shape(token_ids)[1]

        positions = tf.range(sequence_length)

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors


# ==========================================
# 3. TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        self.feed_forward = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, token_mask, training=False):

        attention_mask = token_mask[:, tf.newaxis, :]

        attention_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask,
            training=training
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        x = self.norm1(inputs + attention_output)

        ff_output = self.feed_forward(x)

        ff_output = self.dropout2(
            ff_output,
            training=training
        )

        return self.norm2(x + ff_output)


# ==========================================
# 4. COMPLETE MODEL
# ==========================================

class SentimentTransformer(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.embedding = TokenAndPositionEmbedding(
            VOCAB_SIZE,
            MAX_LENGTH,
            EMBED_DIM
        )

        self.transformer = TransformerBlock(
            EMBED_DIM,
            NUM_HEADS,
            FF_DIM
        )

        self.classifier = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])

    def call(self, token_ids, training=False):

        token_mask = tf.not_equal(token_ids, 0)

        x = self.embedding(token_ids)

        x = self.transformer(
            x,
            token_mask=token_mask,
            training=training
        )

        mask = tf.cast(token_mask, x.dtype)
        mask = mask[:, :, tf.newaxis]

        summed_vectors = tf.reduce_sum(x * mask, axis=1)

        token_counts = tf.reduce_sum(mask, axis=1)
        token_counts = tf.maximum(token_counts, 1.0)

        pooled = summed_vectors / token_counts

        return self.classifier(pooled)


# ==========================================
# 5. RECREATE MODEL AND LOAD WEIGHTS
# ==========================================

model = SentimentTransformer()

# Build all model layers before loading weights.
model(tf.zeros((1, MAX_LENGTH), dtype=tf.int32))

model.load_weights(WEIGHTS_PATH)

print("Saved Transformer loaded successfully!")


# ==========================================
# 6. LOAD THE IMDB WORD VOCABULARY
# ==========================================

from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()


# ==========================================
# 7. PREDICT NEW REVIEWS
# ==========================================

def predict_sentiment(review):

    words = review.lower().split()

    sequence = []

    for word in words:

        index = word_index.get(word)

        if index is None or index + 3 >= VOCAB_SIZE:
            token_id = 2
        else:
            token_id = index + 3

        sequence.append(token_id)

    # IMDB uses 1 as the start-of-review token.
    sequence = [1] + sequence

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    score = float(
        model.predict(padded, verbose=0)[0, 0]
    )

    sentiment = "Positive" if score >= 0.5 else "Negative"

    print(f"\nReview: {review}")
    print(f"Positive score: {score:.3f}")
    print(f"Prediction: {sentiment}")


# ==========================================
# 8. TEST THE SAVED MODEL
# ==========================================

predict_sentiment("This movie was absolutely wonderful")

predict_sentiment("The acting was terrible")

predict_sentiment("I really enjoyed this film")

predict_sentiment("I hated every minute of it")