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

def forward_pass(x, weights, biases, activation_fns, num_layers):
    #store linear transformations zs and activations
    zs = []
    activations = []

    #input to the first layer
    a = x
    for l in range(1,num_layers+1):
        z = np.dot(weights[f"W{l}"], a) + biases[f"b{l}"]
        zs.append(z)
        a = activation_fns[l-1](z)
        activations.append(a)
    return zs, activations

#this is also called the cost of a single training example 
def output_layer_error(actual, predicted):
    return predicted - actual

def backward_pass(x, y, weights, biases, zs, activations, activation_fns, activation_derivative_fns):
    n = len(y)  # Number of examples
    num_layers = len(weights)  # Number of layers in the network
    
    # Initialize dictionaries to store gradients
    grads = {}
    
    # Initialize the error at the output layer (delta for the last layer)
    output_error = activations[-1] - y  # Error in the output layer (aL - y)
    
    # Loop backward through layers
    for l in reversed(range(1, num_layers + 1)):
        # dWl and dbl for current layer
        a_prev = activations[l - 1] if l > 1 else x  # Activation from the previous layer (or input layer)
        z_curr = zs[l - 1]  # Current z (before applying activation)
        
        # Compute gradients for weights and biases
        dWl = (1 / n) * np.dot(output_error, a_prev.T)
        dbl = (1 / n) * np.sum(output_error, axis=1, keepdims=True)
        
        # Store gradients in the dictionary
        grads[f"dW{l}"] = dWl
        grads[f"db{l}"] = dbl
        
        if l > 1:
            # Propagate error backwards
            dz_prev = np.dot(weights[f"W{l}"].T, output_error) * activation_derivative_fns[l-1](z_curr)
            output_error = dz_prev  # Update error for next layer
    
    return grads

def update_parameters(weights, biases, grads, learning_rate):
    for i in range(1, len(weights)+1):
        weights[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
        biases[f"b{i}"] -= learning_rate * grads[f"db{i}"]

    return weights, biases

def train_neural_network(x, y, weights, biases, learning_rate, iterations, activation_fns, activation_derivative_fns):
    num_layers = len(weights)
    for iteration in range(iterations):
        # Forward pass
        zs, activations = forward_pass(x, weights, biases, activation_fns, num_layers)

        a_final = activations[-1]
        
        # Compute loss per sample
        individual_losses = []

        #loop through all samples
        for i in range(len(y)):
            loss = binary_cross_entropy_loss(y[i],a_final[0][i])
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
