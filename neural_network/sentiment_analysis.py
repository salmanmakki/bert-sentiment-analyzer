import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    GlobalAveragePooling1D,
    Dense
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# Small example dataset
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


# Create a vocabulary from our reviews
tokenizer = Tokenizer(
    num_words=1000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(reviews)

# Convert reviews into sequences of integers
sequences = tokenizer.texts_to_sequences(reviews)

# Make every sequence length 6
X = pad_sequences(
    sequences,
    maxlen=6,
    padding="post",
    truncating="post"
)

print("Vocabulary:", tokenizer.word_index)
print("\nFirst review:", reviews[0])
print("Token IDs:", sequences[0])
print("Padded sequence:", X[0])
print("Dataset shape:", X.shape)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential([
    Input(shape=(6,)),

    Embedding(
        input_dim=vocab_size,
        output_dim=8,
        mask_zero=True
    ),

    GlobalAveragePooling1D(),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    X,
    labels,
    epochs=100,
    verbose=0
)

print("Training complete!")

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

    print("\nReview:", review)
    print("Positive score:", round(score, 3))
    print("Prediction:", sentiment)

predict_sentiment("this movie is wonderful")
predict_sentiment("this movie is awful")
predict_sentiment("i love this film")
predict_sentiment("i hate this film")
predict_sentiment("this movie is not good")
predict_sentiment("this movie is not bad")