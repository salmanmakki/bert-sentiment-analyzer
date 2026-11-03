import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# ==========================================
# 1. CONFIGURATION
# ==========================================

VOCAB_SIZE = 10_000
MAX_LENGTH = 200

EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256

BATCH_SIZE = 64
EPOCHS = 3


# ==========================================
# 2. LOAD THE IMDB DATASET
# ==========================================

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

# Keep a smaller subset so the experiment is manageable.
X_validation = X_train[10_000:12_500]
y_validation = y_train[10_000:12_500]

X_train = X_train[:10_000]
y_train = y_train[:10_000]

X_test = X_test[:2_500]
y_test = y_test[:2_500]

# Make all reviews the same length.
X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_validation = pad_sequences(
    X_validation,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("Training data:", X_train.shape)
print("Validation data:", X_validation.shape)
print("Test data:", X_test.shape)


# ==========================================
# 3. TOKEN + POSITIONAL EMBEDDINGS
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

        positions = tf.range(
            start=0,
            limit=sequence_length,
            delta=1
        )

        token_vectors = self.token_embedding(token_ids)

        position_vectors = self.position_embedding(positions)

        return token_vectors + position_vectors


# ==========================================
# 4. TRANSFORMER BLOCK
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

        # Self-attention ignores padding tokens as keys.
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

        # Residual connection and normalization.
        x = self.norm1(inputs + attention_output)

        # Feed-forward network.
        ff_output = self.feed_forward(x)

        ff_output = self.dropout2(
            ff_output,
            training=training
        )

        # Second residual connection and normalization.
        return self.norm2(x + ff_output)


# ==========================================
# 5. COMPLETE SENTIMENT TRANSFORMER
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

        # True for real tokens; False for padding.
        token_mask = tf.not_equal(token_ids, 0)

        # Convert token IDs into contextual input vectors.
        x = self.embedding(token_ids)

        # Add Transformer processing.
        x = self.transformer(
            x,
            token_mask=token_mask,
            training=training
        )

        # Ignore padded positions during average pooling.
        mask = tf.cast(token_mask, x.dtype)
        mask = mask[:, :, tf.newaxis]

        summed_vectors = tf.reduce_sum(x * mask, axis=1)

        token_counts = tf.reduce_sum(mask, axis=1)
        token_counts = tf.maximum(token_counts, 1.0)

        pooled = summed_vectors / token_counts

        # Predict positive sentiment.
        return self.classifier(pooled)


# ==========================================
# 6. COMPILE THE MODEL
# ==========================================

model = SentimentTransformer()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Build the model before displaying its summary.
model(tf.zeros((1, MAX_LENGTH), dtype=tf.int32))

model.summary()


# ==========================================
# 7. TRAIN THE MODEL
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=10,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)


# ==========================================
# 8. EVALUATE ON UNSEEN REVIEWS
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

# Save the trained model
model.save_weights("transformer_sentiment.weights.h5")

print("\nModel weights saved successfully!")

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")

# Get predictions for the test reviews
scores = model.predict(X_test, verbose=0).flatten()

# Convert scores into classes
y_pred = (scores >= 0.5).astype(int)

# Compare predictions with the correct labels
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Show precision, recall and F1-score
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        zero_division=0
    )
)

# Visualize the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot()
plt.title("Transformer Sentiment Classification")
plt.tight_layout()
plt.show()


# ==========================================
# 9. PREDICT NEW REVIEWS
# ==========================================

word_index = imdb.get_word_index()

reverse_word_index = {
    index + 3: word
    for word, index in word_index.items()
}


def predict_sentiment(review):

    words = review.lower().split()

    sequence = []

    for word in words:
        # Look up the word in the IMDB vocabulary.
        index = word_index.get(word)

        # Convert vocabulary indices to IMDB token IDs.
        if index is None or index + 3 >= VOCAB_SIZE:
            token_id = 2
        else:
            token_id = index + 3

        sequence.append(token_id)

    # Add the start-of-review token.
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

    prediction = (
        "Positive" if score >= 0.5 else "Negative"
    )

    print(f"\nReview: {review}")
    print(f"Positive score: {score:.3f}")
    print(f"Prediction: {prediction}")

predict_sentiment("This movie was absolutely wonderful")
predict_sentiment("The acting was terrible")
predict_sentiment("I really enjoyed this film")
predict_sentiment("I hated every minute of it")


# Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)

plt.show()


# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Transformer: Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.show()