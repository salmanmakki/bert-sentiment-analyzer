import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences


# ==========================================
# 1. OUR TRAINING DATASET
# ==========================================

reviews = [
    # Positive reviews
    "this movie is amazing",
    "i love this movie",
    "this film is wonderful",
    "what a fantastic movie",
    "the movie was excellent",
    "i really enjoyed this film",
    "this movie is good",
    "the film was great",
    "i liked this movie",
    "this was a brilliant film",
    "the movie was very good",
    "i really love this film",

    # Negative reviews
    "this movie is terrible",
    "i hate this movie",
    "this film is boring",
    "what a horrible movie",
    "the movie was awful",
    "i really disliked this film",
    "this movie is bad",
    "the film was terrible",
    "i did not like this movie",
    "this was a disappointing film",
    "the movie was very bad",
    "i really hate this film",

    # More examples involving negation
    "this movie is not good",
    "this film is not great",
    "i do not love this movie",
    "this movie was not enjoyable",
    "the film is not amazing",
    "this was not a good movie",

    # Positive meanings involving negation
    "this movie is not bad",
    "the film was not terrible",
    "i do not hate this movie",
    "this movie was not awful"
]

labels = np.array(
    [1] * 12 + [0] * 12 + [0] * 6 + [1] * 4,
    dtype=np.float32
)


# ==========================================
# 2. CONVERT WORDS INTO NUMBERS
# ==========================================

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

sequences = tokenizer.texts_to_sequences(reviews)

X = pad_sequences(
    sequences,
    maxlen=6,
    padding="post",
    truncating="post"
)

y = labels

vocab_size = len(tokenizer.word_index) + 1

print("Vocabulary size:", vocab_size)
print("Input shape:", X.shape)


# ==========================================
# 3. BUILD THE LSTM MODEL
# ==========================================

model = Sequential([
    Input(shape=(6,)),

    Embedding(
        input_dim=vocab_size,
        output_dim=16,
        mask_zero=True
    ),

    LSTM(16),

    Dense(1, activation="sigmoid")
])


# ==========================================
# 4. COMPILE THE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 5. TRAIN THE MODEL
# ==========================================

history = model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

print("\nTraining completed!")

# Evaluate on the training examples
loss, accuracy = model.evaluate(X, y, verbose=0)

print(f"\nTraining loss: {loss:.4f}")
print(f"Training accuracy: {accuracy:.2%}")

# Check the model's prediction for every training example
scores = model.predict(X, verbose=0).flatten()

predictions = (scores >= 0.5).astype(int)

correct = np.sum(predictions == y)

print(f"\nCorrect training predictions: {correct}/{len(y)}")

# Display a few individual examples
for i in range(5):
    print(
        f"Review: {reviews[i]:30s} | "
        f"Actual: {int(y[i])} | "
        f"Predicted score: {scores[i]:.3f} | "
        f"Predicted class: {predictions[i]}"
    )


# ==========================================
# 6. PREDICT SENTIMENT
# ==========================================

def predict_sentiment(review):

    sequence = tokenizer.texts_to_sequences([review])

    padded = pad_sequences(
        sequence,
        maxlen=6,
        padding="post",
        truncating="post"
    )

    score = float(
        model.predict(padded, verbose=0)[0, 0]
    )

    sentiment = "Positive" if score >= 0.5 else "Negative"

    print("\nReview:", review)
    print(f"Positive score: {score:.3f}")
    print("Prediction:", sentiment)


# ==========================================
# 7. TEST OUR DETECTIVE
# ==========================================

predict_sentiment("this movie is amazing")
predict_sentiment("this movie is terrible")
predict_sentiment("this movie is not good")
predict_sentiment("this movie is not bad")
predict_sentiment("i love this film")
predict_sentiment("i hate this film")