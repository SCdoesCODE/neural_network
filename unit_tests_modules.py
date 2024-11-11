from src.utils import sigmoid, sigmoid_derivative, relu, relu_derivative, softmax, binary_cross_entropy_loss
import numpy as np 
from keras.losses import binary_crossentropy

#source: https://www.mathcelebrity.com/sigmoid-function-calculator.php?num=-1&pl=Calculate+Sigmoid
np.testing.assert_allclose(sigmoid(np.array([0, 1, -1])), np.array([0.5, 0.73105857863, 0.26894142137]), rtol=1e-5)

#source: https://www.redcrab-software.com/en/Calculator/Derivative-Sigmoid
np.testing.assert_allclose(sigmoid_derivative(np.array([0, 1, -1])), np.array([0.25, 0.196612, 0.196612]), rtol=1e-5)

#source: https://www.vcalc.com/wiki/vCalc/ReLu
np.testing.assert_allclose(relu(np.array([0, 1, -1])), np.array([0, 1, 0]), rtol=1e-5)

#source: understanding of relu function, e.g. this graph: https://aman.ai/primers/backprop/assets/relu/relu.jpg
np.testing.assert_allclose(relu_derivative(np.array([0, 1, -1])), np.array([0, 1, 0]), rtol=1e-5)

#source: https://superbotz.com/maths/functions-special-functions/softmax-function/index.php
np.testing.assert_allclose(softmax(np.array([1,2,3,4,5])), np.array([0.01165623095604, 0.031684920796124, 0.086128544436269, 0.23412165725274, 0.63640864655883]), rtol=1e-5)

# Example true labels and predicted probabilities from source: https://www.geeksforgeeks.org/binary-cross-entropy-log-loss-for-binary-classification/
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
bce_loss_from_keras = binary_crossentropy(y_true, y_pred)

np.testing.assert_allclose(binary_cross_entropy_loss(y_true,y_pred), bce_loss_from_keras, rtol=1e-5)




