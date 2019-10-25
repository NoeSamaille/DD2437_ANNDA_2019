import numpy as np


def residual_err(predictions, targets):
    return np.average(np.absolute(predictions - targets))


def pcn_predict(patterns, v, w):
    # Forward pass threw the network
    # Compute hidden layer input
    h_in = np.dot(v, patterns)
    # Compute hidden layer output
    h_out = (2 / (1 + np.exp(-1 * h_in))) - 1
    # Add the bias for the hidden layer
    h_out = np.concatenate((h_out, np.ones((1, len(patterns[0])))))
    # Compute output layer input
    y_in = np.dot(w, h_out)
    # Compute output layer output (= NN output)
    y_out = (2 / (1 + np.exp(-1 * y_in))) - 1
    return y_out, h_out


def pcn_train(patterns, targets, nb_epochs, l_rate, nb_hidden_nodes, nb_input_nodes=2, nb_output_nodes=1, init_weights_sd=1):
    # Init the weights
    v = np.random.normal(0, init_weights_sd, (nb_hidden_nodes, nb_input_nodes + 1))
    w = np.random.normal(0, init_weights_sd, (nb_output_nodes, nb_hidden_nodes + 1))
    # Store the transposed patterns
    t_patterns = np.transpose(patterns)
    # First deltas are 0
    dv, dw = 0, 0
    error = np.zeros(nb_epochs)
    # Loop over the epochs
    for epoch in range(nb_epochs):
        # Forward pass
        y_out, h_out = pcn_predict(patterns, v, w)
        # Backward pass
        delta_y = (y_out - targets) * (((1 + y_out) * (1 - y_out)) * 0.5)
        delta_h = np.dot(np.transpose(w), delta_y) * (((1 + h_out) * (1 - h_out)) * 0.5)
        delta_h = delta_h[:-1] # Remove the bias from the forward pass
        # Weight update
        alpha = 0.9 # Momentum factor
        dv = (dv * alpha) + (1 - alpha) * np.dot(delta_h, t_patterns)
        dw = (dw * alpha) + (1 - alpha) * np.dot(delta_y, np.transpose(h_out))
        v = v - l_rate * dv
        w = w - l_rate * dw
        error[epoch] = residual_err(y_out, targets)
    return v, w, error
