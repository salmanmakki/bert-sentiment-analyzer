import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", X_test.shape)
print("Test labels:", y_test.shape)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())

# Create the neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make a prediction for one image
prediction = model.predict(X_test[0:1])

print("Prediction probabilities:")
print(prediction)

predicted_digit = prediction.argmax()

print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[0])


plt.imshow(X_test[0], cmap="gray")
plt.title(f"Predicted: {predicted_digit} | Actual: {y_test[0]}")
plt.axis("off")
plt.show()

# Show the first image
plt.imshow(X_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()

# Get the first Dense layer
weights, biases = model.layers[1].get_weights()

print("Weights shape:", weights.shape)
print("Biases shape:", biases.shape)

# Take one test image
image = X_test[0]

print("Image shape:", image.shape)


image_flat = image.flatten()

print("Flattened image shape:", image_flat.shape)

# Calculate neuron 0 manually
neuron_0 = np.dot(image_flat, weights[:, 0]) + biases[0]

print("Neuron 0 before ReLU:", neuron_0)