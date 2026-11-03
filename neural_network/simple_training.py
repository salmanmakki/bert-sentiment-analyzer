import numpy as np
from sklearn.linear_model import LogisticRegression

# Our training examples
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

# Correct answers
y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# Make a prediction
prediction = model.predict([[5]])

print("Prediction:", prediction[0])