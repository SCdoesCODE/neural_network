#testing the neural network modules on the simple XOR problem

from src.utils import train_neural_network, relu, sigmoid, relu_derivative, sigmoid_derivative, forward_pass
from sklearn.neural_network import MLPClassifier
import numpy as np

def validate_inputs(x, y):
    """
    Validates the inputs x and y to ensure they have the correct shapes and structures.

    Args:
        x (np.ndarray): Input features, expected shape (num_features, batch_size).
        y (np.ndarray): Target values, expected shape (num_outputs, batch_size).

    Raises:
        AssertionError: If any input does not meet the required conditions.
    """
    # Check if x and y are numpy arrays
    assert isinstance(x, np.ndarray), f"Input x should be a numpy array, got {type(x)}."
    assert isinstance(y, np.ndarray), f"Input y should be a numpy array, got {type(y)}."

    # Check the dimensions of x
    assert x.ndim == 2, f"Input x should have 2 dimensions (num_features, batch_size), got {x.ndim}."

    # Check the dimensions of y
    assert y.ndim == 2, f"Input y should have 2 dimensions (num_outputs, batch_size), got {y.ndim}."

    # Check that the batch size matches in x and y
    assert x.shape[1] == y.shape[1], (
        f"Batch size mismatch between x and y. x has {x.shape[1]} examples, "
        f"but y has {y.shape[1]} examples."
    )


def he_initialization(size_in = 2, size_out = 2, random_seed = 42):
    np.random.seed(random_seed)
    return np.random.randn(size_out, size_in) * np.sqrt(2 / size_in)

def input_XOR():
    # XOR inputs and outputs
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0, 1, 1, 0]])  # Expected outputs
    return x,y

def initialize_weights_one_hidden_layer():
    weights = {
    "W1": he_initialization(size_in=2, size_out=2),  # 2 neurons, 2 inputs
    "W2": he_initialization(size_in=2, size_out=1)  # 1 neuron, 2 inputs
    }
    biases = {
        "b1": np.ones((2, 1)),
        "b2": np.ones((1, 1))
    }
    return weights, biases

def initialize_weights_two_hidden_layers(random_seed = 42):
    weights = {
        "W1": he_initialization(size_in=2, size_out=4, random_seed = random_seed),  # 4 neurons in the first hidden layer
        "W2": he_initialization(size_in=4, size_out=2, random_seed = random_seed),  # 2 neurons in the second hidden layer
        "W3": he_initialization(size_in=2, size_out=1, random_seed = random_seed)   # 1 neuron in the output layer
    }
    biases = {
        "b1": np.ones((4, 1)),  # Bias for the first hidden layer
        "b2": np.ones((2, 1)),  # Bias for the second hidden layer
        "b3": np.ones((1, 1))   # Bias for the output layer
    }
    return weights, biases

def initialize_weights_three_hidden_layers():
    weights = {
        "W1": he_initialization(size_in=4, size_out=8),  # 8 neurons in the first hidden layer
        "W2": he_initialization(size_in=8, size_out=6),  # 6 neurons in the second hidden layer
        "W3": he_initialization(size_in=6, size_out=4),  # 4 neurons in the third hidden layer
        "W4": he_initialization(size_in=4, size_out=1)   # 1 neuron in the output layer
    }
    biases = {
        "b1": np.ones((8, 1)),  # Bias for the first hidden layer
        "b2": np.ones((6, 1)),  # Bias for the second hidden layer
        "b3": np.ones((4, 1)),  # Bias for the third hidden layer
        "b4": np.ones((1, 1))   # Bias for the output layer
    }
    return weights, biases

def train(x,y,weights,biases, lr, activation_fns, activations_derivative_fns):
    trained_weights, trained_biases = train_neural_network(
        x, y, weights, biases,
        learning_rate=lr,
        iterations=10000,
        activation_fns=activation_fns,
        activation_derivative_fns=activations_derivative_fns
    )
    return trained_weights,trained_biases

def predict(x,trained_weights, trained_biases):
    _, activations = forward_pass(x, trained_weights, trained_biases, [relu, sigmoid])
    predictions = (activations[-1] > 0.5).astype(int)  # Threshold at 0.5
    return predictions


for i in range(100):
    x,y = input_XOR()
    #weights, biases = initialize_weights_one_hidden_layer()
    random_seed = i
    weights, biases = initialize_weights_two_hidden_layers(random_seed)
    #weights, biases = initialize_weights_three_hidden_layers()
    trained_weights, trained_biases = train(x,y,weights,biases, 0.01, [relu, relu, sigmoid], [relu_derivative, relu_derivative, sigmoid_derivative])
    predictions = predict(x,trained_weights, trained_biases)
    expected_predictions = y  # XOR ground truth
    # Test if predictions are correct
    try:
        np.testing.assert_array_equal(predictions, expected_predictions)
        print(f"Found correct solution at iteration {i} with seed {seed}")
        break  # Exit the loop if the solution is found
    except AssertionError:
        pass  # Continue if the prediction is incorrect



x,y = input_XOR()
expected_predictions = y

print("MLPClassifier")

# Create a simple neural network with one hidden layer
model = MLPClassifier(hidden_layer_sizes=(4,2,1), activation = "relu", learning_rate = "constant", learning_rate_init = 0.01, max_iter = 10000, random_state=42)

# Train the model
model.fit(x.T, y.ravel())

# Make predictions
predictions = model.predict(x.T)

np.testing.assert_array_equal(predictions, expected_predictions[0])

"""
Some comments:
- with one hidden layer the network is SUPER sensitive to the initialisation of the weights (so the random seed number in this case)
- I tried different learning rates, iteration numbers without success with the one hidden layer containing two nodes
"""