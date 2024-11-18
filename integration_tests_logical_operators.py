#testing the neural network modules on the simple XOR problem

from src.utils import train_neural_network, relu, sigmoid, relu_derivative, sigmoid_derivative, forward_pass
from sklearn.neural_network import MLPClassifier
import numpy as np


def he_initialization(size_in, size_out):
    np.random.seed(42)
    return np.random.randn(size_out, size_in) * np.sqrt(2 / size_in)

def input_XOR():
    # XOR inputs and outputs
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0, 1, 1, 1]])  # Expected outputs
    return x,y

def initialize_weights():
    weights = {
    "W1": he_initialization(size_in=2, size_out=2),  # 2 neurons, 2 inputs
    "W2": he_initialization(size_in=2, size_out=1)  # 1 neuron, 2 inputs
    }
    biases = {
        "b1": np.ones((2, 1)),
        "b2": np.ones((1, 1))
    }
    return weights, biases

def train(x,y,weights,biases, lr):
    trained_weights, trained_biases = train_neural_network(
        x, y, weights, biases,
        learning_rate=lr,
        iterations=10000,
        activation_fns=[relu, sigmoid],
        activation_derivative_fns=[relu_derivative, sigmoid_derivative]
    )
    return trained_weights,trained_biases

def predict(trained_weights, trained_biases):
    _, activations = forward_pass(x, trained_weights, trained_biases, [relu, sigmoid], 2)
    predictions = (activations[-1] > 0.5).astype(int)  # Threshold at 0.5
    return predictions

x,y = input_XOR()
weights, biases = initialize_weights()
trained_weights, trained_biases = train(x,y,weights,biases, lr = 0.1)
predictions = predict(trained_weights, trained_biases)
expected_predictions = y  # XOR ground truth
np.testing.assert_array_equal(predictions, expected_predictions)

# Create a simple neural network with one hidden layer
model = MLPClassifier(hidden_layer_sizes=(2,1), activation = "relu", learning_rate = "constant", learning_rate_init = 0.1, max_iter = 10000, random_state=42)

# Train the model
model.fit(x.T, y.ravel())

# Make predictions
predictions = model.predict(x.T)

np.testing.assert_array_equal(predictions, expected_predictions[0])

"""
Some comments after finally getting it to work with XOR:
- what finally made it work was changing biases to 1s instead of 0s
- a learning rate of 0.01 is too big, it worked when I changed it to 0.1
- 1000 iterations e.g. is not enough, it worked when I changed it to 10000
- the loss barely moves, I don´t think I´m super satisfied with this
- the activations are also quite close to 0.5 - I would like them to be closer to either 0 or 1, at this point I almost feel like we´re guessing
"""