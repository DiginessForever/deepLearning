import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
print("input (x): " + str(x))
print("weights_input_hidden: \n" + str(weights_input_hidden))

hidden_layer_input = np.dot(x, weights_input_hidden)
print("hidden layer before sigmoid (hidden_layer_input): " + str(hidden_layer_input))

hidden_layer_output = sigmoid(hidden_layer_input)
print("hidden layer after sigmoid (hidden_layer_output): " + str(hidden_layer_output))
print("weights_hidden_output: " + str(weights_hidden_output))

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
print("output_layer_in: " + str(output_layer_in))

output = sigmoid(output_layer_in)
print("output layer after signmoid (output): " + str(output))

## Backwards pass
## TODO: Calculate output error
error = target - output
deriv_output = output * (1 - output)

# TODO: Calculate error term for output layer
output_error_term = error * deriv_output
print("output_error_term, sig(out) * (1-sig(out)): " + str(output_error_term))

# TODO: Calculate error term for hidden layer
#l1_delta = error * deriv_output

hidden_error_term = weights_hidden_output * output_error_term * hidden_layer_output * (1 - hidden_layer_output)
print("hidden error term (weights_hidden_output * output_error_term * hidden_layer_output \n* (1 - hidden_layer_output): \n" + str(hidden_error_term))

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:,None]

print('Change in weights for hidden layer to output layer (delta_w_h_o): ' + str(delta_w_h_o))
print('Change in weights for input layer to hidden layer (delta_w_i_h): ' + str(delta_w_i_h))
