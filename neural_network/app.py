from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR.parent / "transformer_sentiment.weights.h5"

VOCAB_SIZE = 10_000
MAX_LENGTH = 200
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, vocab_size, max_length, embed_dim):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, embed_dim)
        self.position_embedding = layers.Embedding(max_length, embed_dim)

    def call(self, token_ids):
        sequence_length = tf.shape(token_ids)[1]
        positions = tf.range(sequence_length)
        return self.token_embedding(token_ids) + self.position_embedding(positions)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
        )
        self.feed_forward = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
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
            training=training,
        )
        attention_output = self.dropout1(attention_output, training=training)
        x = self.norm1(inputs + attention_output)
        feed_forward_output = self.dropout2(
            self.feed_forward(x),
            training=training,
        )
        return self.norm2(x + feed_forward_output)


class SentimentTransformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(
            VOCAB_SIZE,
            MAX_LENGTH,
            EMBED_DIM,
        )
        self.transformer = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
        self.classifier = tf.keras.Sequential(
            [
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def call(self, token_ids, training=False):
        token_mask = tf.not_equal(token_ids, 0)
        x = self.transformer(
            self.embedding(token_ids),
            token_mask=token_mask,
            training=training,
        )
        mask = tf.cast(token_mask[:, :, tf.newaxis], x.dtype)
        summed_vectors = tf.reduce_sum(x * mask, axis=1)
        token_counts = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        return self.classifier(summed_vectors / token_counts)


if not WEIGHTS_PATH.is_file():
    raise FileNotFoundError(
        f"Model weights not found: {WEIGHTS_PATH}. "
        "Place transformer_sentiment.weights.h5 in the project directory."
    )

model = SentimentTransformer()
model(tf.zeros((1, MAX_LENGTH), dtype=tf.int32))
model.load_weights(WEIGHTS_PATH)
word_index = imdb.get_word_index()

app = FastAPI(title="Sentiment API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewRequest(BaseModel):
    text: str


def review_to_sequence(text: str):
    sequence = [1]
    for word in text.lower().split():
        index = word_index.get(word)
        sequence.append(2 if index is None or index + 3 >= VOCAB_SIZE else index + 3)
    return pad_sequences(
        [sequence],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post",
    )


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    text = request.text.strip()
    if not text:
        return {"error": "Review cannot be empty"}

    score = float(model(review_to_sequence(text), training=False)[0, 0])
    return {
        "sentiment": "Positive" if score >= 0.5 else "Negative",
        "confidence": round((score if score >= 0.5 else 1 - score) * 100, 2),
    }


@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )