import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    SimpleRNN,
    Dense
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# Example training data
reviews = [
    "this movie is amazing",
    "i love this movie",
    "this film is wonderful",
    "what a fantastic movie",
    "the movie was excellent",
    "i really enjoyed this film",
    "this movie is terrible",
    "i hate this movie",
    "this film is boring",
    "what a horrible movie",
    "the movie was awful",
    "i really disliked this film"
]

# 1 = positive, 0 = negative
labels = np.array([
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0
], dtype=np.float32)

# Tokenize text
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)

X = pad_sequences(
    sequences,
    maxlen=6,
    padding="post",
    truncating="post"
)

vocab_size = len(tokenizer.word_index) + 1

# Build the RNN model
model = Sequential([
    Input(shape=(6,)),
    Embedding(
        input_dim=vocab_size,
        output_dim=8,
        mask_zero=True
    ),
    SimpleRNN(16),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X,
    labels,
    epochs=100,
    verbose=0
)

# Prediction function
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=6,
        padding="post",
        truncating="post"
    )

    score = float(model.predict(padded, verbose=0)[0, 0])

    sentiment = "Positive" if score >= 0.5 else "Negative"

    print(f"\nReview: {review}")
    print(f"Positive score: {score:.3f}")
    print(f"Prediction: {sentiment}")

# Try different reviews
predict_sentiment("this movie is amazing")
predict_sentiment("this movie is terrible")
predict_sentiment("this movie is not good")