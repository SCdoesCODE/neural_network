# Neural Networks

Refreshing my memory in Neural Network land

## Notebooks

Initial work for this repo before turning into modules and utils functions

### neural_network_from_scratch.ipynb
- an example of a 2-layer multilayer perceptron or neural network
- each function only works with this particular type of network

### neural_network_from_scratch_general.ipynb
- expands the neural_network_from_scratch.ipynb notebook to work for feedforward networks of any size

## Utils Functions

### src/utils.py

Contains more updated and cleaner versions of functions developed in **neural_network_from_scratch_general.ipynb**. Contains functions needed to train and test a neural network such as e.g. the relu or sigmoid functions. Contains functions for the forward and backward passes used when training neural networks. 

## Testing

### integration_tests_logical_operators.py

Currently contains one integration test running through all functions in **src/utils.py** to train and test the neural network on the simple XOR problem (logical operator). The results are also compared against the results given by the MLPClassifier (Sklearn). 

### unit_tests_modules.py

Unit tests run against computations made on different calculators found online. Unit tests are run on functions such as the sigmoid function, the derivative of the sigmoid function, the relu function, the binary cross entropy loss function etc. 



