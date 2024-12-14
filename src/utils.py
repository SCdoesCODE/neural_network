#let´s start out by gathering all modules/functions here before we start organising them into different packages
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of z.

    Args:
        z (np.ndarray): The input array for which to compute the sigmoid.

    Returns:
        np.ndarray: The sigmoid of each element in z.

    Examples:
        >>> sigmoid(np.array([0, 1, -1]))
        array([0.5, 0.731, 0.269])
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the sigmoid function.

    Args:
        z (np.ndarray): The input array for which to compute the derivative of sigmoid.

    Returns:
        np.ndarray: The derivative of the sigmoid function at each element in z.

    Examples:
        >>> sigmoid_derivative(np.array([0, 1, -1]))
        array([0.25, 0.196, 0.196])
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the ReLU (Rectified Linear Unit) of x.

    Args:
        x (np.ndarray): The input array for which to compute the ReLU.

    Returns:
        np.ndarray: The ReLU of each element in x (non-negative values).

    Examples:
        >>> relu(np.array([-2, 3, 0]))
        array([0, 3, 0])
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of the ReLU function.

    Args:
        x (np.ndarray): The input array for which to compute the derivative of ReLU.

    Returns:
        np.ndarray: The derivative of the ReLU function at each element in x.
                    Returns 1 if x > 0, otherwise 0.

    Examples:
        >>> relu_derivative(np.array([-2, 3, 0]))
        array([0, 1, 0])
    """
    return (x > 0).astype(float)


def softmax(vector: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a vector.

    Args:
        vector (np.ndarray): A 1D array of values.

    Returns:
        np.ndarray: The softmax of the input vector, with values normalized to sum to 1.

    Examples:
        >>> softmax(np.array([1, 2, 3]))
        array([0.090, 0.244, 0.665])
    """
    shift_vector = vector - np.max(vector)
    exp_values = np.exp(shift_vector)
    return exp_values / np.sum(exp_values)

#binary cross entropy loss for a single data point
def binary_cross_entropy_loss(y_true, y_predicted):
    # Clip predicted value to avoid log(0)
    very_small_number = 1e-7
    y_predicted = np.clip(y_predicted, very_small_number, 1 - very_small_number)
    
    # Calculate binary cross-entropy for a single point
    loss = - (y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted))
    
    # Return scalar if inputs are arrays by taking the mean
    return np.mean(loss) if isinstance(loss, np.ndarray) else loss

def forward_pass(x, weights, biases, activation_fns):
    #store linear transformations zs and activations
    num_layers = len(weights)
    zs = []
    activations = []

    #input to the first layer
    a = x
    for l in range(1,num_layers+1):
        z = np.dot(weights[f"W{l}"], a) + biases[f"b{l}"]

        # Assert shapes
        assert z.shape == (weights[f"W{l}"].shape[0], a.shape[1]), (
            f"Layer {l}: z shape mismatch. "
            f"z is computed as W.dot(a_prev) + b, so it should be (neurons_in_current_layer, batch_size). "
            f"Got {z.shape}, expected {(weights[f'W{l}'].shape[0], a.shape[1])}."
        )

        zs.append(z)
        a = activation_fns[l-1](z)

        # Assert shape of activation
        assert a.shape == z.shape, (
            f"Layer {l}: Activation shape mismatch. "
            f"The activation function is applied element-wise, so it should return a matrix of the same shape as z. "
            f"Got {a.shape}, expected {z.shape}."
        )
        
        activations.append(a)
    return zs, activations

#this is also called the cost of a single training example 
def output_layer_error(actual, predicted):
    return predicted - actual

def backward_pass(x, y, weights, biases, zs, activations, activation_fns, activation_derivative_fns):
    n = y.size  # Number of examples
    num_layers = len(weights)  # Number of layers in the network

    # Initialize dictionaries to store gradients
    grads = {}
    
    # Initialize the error at the output layer (delta for the last layer)
    output_error = activations[-1] - y  # Error in the output layer (aL - y)

    assert output_error.shape == activations[-1].shape, (
        f"Output error shape mismatch. "
        f"The output error is computed as activations[-1] - y, so it should match the shape of the output layer's activations "
        f"(output_size, batch_size). Got {output_error.shape}, expected {activations[-1].shape}."
    )
    # Loop backward through layers
    for l in reversed(range(1,num_layers+1)):
        # dWl and dbl for current layer
        a_prev = activations[l - 2] if l > 1 else x  # Activation from the previous layer (or input layer)

        # Assert that a_prev has the correct shape
        if l > 1:
            # For layers after the first, a_prev should have shape (neurons_in_previous_layer, batch_size)
            assert a_prev.shape == (weights[f"W{l}"].shape[1], x.shape[1]), f"Layer {l}: a_prev has incorrect shape. Expected {(weights[f'W{l}'].shape[1], x.shape[1])}, but got {a_prev.shape}"
        else:
            # For the first layer, a_prev (input x) should have shape (input_size, batch_size)
            assert a_prev.shape == (x.shape[0], x.shape[1]), f"Layer {l}: a_prev (input) has incorrect shape. Expected {(x.shape[0], x.shape[1])}, but got {a_prev.shape}"

        z_curr = zs[l-2]  # Current z (before applying activation)

        """# Assert that z_curr shape matches the output layer's z shape
        assert z_curr.shape == (weights[f"W{num_layers}"].shape[0], x.shape[1]), (
            f"Layer {num_layers}: z_curr has incorrect shape. Expected "
            f"({weights[f'W{num_layers}'].shape[0]}, {x.shape[1]}), but got {z_curr.shape}."
        )"""
        
        # Compute gradients for weights and biases
        dWl = (1 / n) * np.dot(output_error, a_prev.T)
        dbl = (1 / n) * np.sum(output_error, axis=1, keepdims=True)

        # Assert gradient shapes
        assert dWl.shape == weights[f"W{l}"].shape, (
            f"Layer {l}: dW shape mismatch. "
            f"dW is computed as (1/batch_size) * output_error.dot(a_prev.T), so it should match the shape of weights['W{l}'] "
            f"which is (neurons_in_current_layer, neurons_in_previous_layer). "
            f"Got {dWl.shape}, expected {weights[f'W{l}'].shape}."
        )
        assert dbl.shape == biases[f"b{l}"].shape, (
            f"Layer {l}: db shape mismatch. "
            f"db is computed as (1/batch_size) * sum(output_error, axis=1), so it should match the shape of biases['b{l}'] "
            f"which is (neurons_in_current_layer, 1). "
            f"Got {dbl.shape}, expected {biases[f'b{l}'].shape}."
        )
        
        # Store gradients in the dictionary
        grads[f"dW{l}"] = dWl
        grads[f"db{l}"] = dbl
        
        if l > 1:
            # Propagate error backwards
            dz_prev = np.dot(weights[f"W{l}"].T, output_error) * activation_derivative_fns[l-1](z_curr)

            # Assert propagated error shape
            assert dz_prev.shape == z_curr.shape, (
                f"Layer {l}: dz_prev shape mismatch. "
                f"dz_prev is computed as W.T.dot(output_error) * activation_derivative(z_curr), so it should have the same shape as z_curr, "
                f"which is (neurons_in_current_layer, batch_size). "
                f"Got {dz_prev.shape}, expected {z_curr.shape}."
            )

            output_error = dz_prev # Update error for next layer
    
    return grads

def update_weights(weights, biases, grads, learning_rate):
    for i in range(1, len(weights)+1):
        weights[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
        biases[f"b{i}"] -= learning_rate * grads[f"db{i}"]

    return weights, biases

def update_parameters(weights, biases, grads, learning_rate):
    for l in range(1, len(weights) + 1):
        W_update = grads[f"dW{l}"]
        b_update = grads[f"db{l}"]
        
        # Assert updates match weight and bias shapes
        assert W_update.shape == weights[f"W{l}"].shape, (
            f"Layer {l}: Weight update shape mismatch. "
            f"The update for W{l} (learning_rate * dW) must match the shape of weights['W{l}'], "
            f"which is (neurons_in_current_layer, neurons_in_previous_layer). "
            f"Got {W_update.shape}, expected {weights[f'W{l}'].shape}."
        )
        assert b_update.shape == biases[f"b{l}"].shape, (
            f"Layer {l}: Bias update shape mismatch. "
            f"The update for b{l} (learning_rate * db) must match the shape of biases['b{l}'], "
            f"which is (neurons_in_current_layer, 1). "
            f"Got {b_update.shape}, expected {biases[f'b{l}'].shape}."
        )

        weights[f"W{l}"] -= learning_rate * W_update
        biases[f"b{l}"] -= learning_rate * b_update

    return weights, biases

def train_neural_network(x, y, weights, biases, learning_rate, iterations, activation_fns, activation_derivative_fns):

    for iteration in range(iterations):
        # Forward pass
        zs, activations = forward_pass(x, weights, biases, activation_fns)

        a_final = activations[-1]
        
        # Compute loss per sample
        individual_losses = []

        #loop through all samples
        for i, yi in np.ndenumerate(y):
            loss = binary_cross_entropy_loss(yi,a_final[i])
            individual_losses.append(loss)

        # Calculate the average loss after each iteration
        avg_loss = np.mean(individual_losses)
        if iteration % 1000 == 0:  # Print every 1000 iterations for example
            print(f"Iteration {iteration}, Average Loss: {avg_loss}")
        
        # Backward pass
        grads = backward_pass(x, y, weights, biases, zs, activations, activation_fns, activation_derivative_fns)
        
        # Update parameters
        weights, biases = update_parameters(weights, biases, grads, learning_rate)
    
    return weights, biases
