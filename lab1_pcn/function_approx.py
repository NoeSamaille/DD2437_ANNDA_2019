#!/bin/bash

#Approximation function using two-layer perceptron

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from two_layer_pcn import predict
from drawnow import drawnow


def mse(y_out, targets):
    error = targets - y_out
    return np.sum(error**2) / len(targets)


def plot_3D():
    """
    Example of how to do a 3D plot, see plot_function for the real implementation
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.mgrid[-10:10:1, -10:10:1]
    Z = np.exp(-(X ** 2 + Y ** 2) / 10 - 0.5)
    ax.plot_wireframe(X, Y, Z)
    plt.show()


def plot_function(fig, X, Y, Z, title):
    """
    plots approximated function
    :param X:
    :param Y:
    :param Z:
    :return:
    """
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    plt.title(title)


def generate_data(start, end, step):
    """
    generates X, Y square grid and reshapes it in vectors of size (1, n*m)
    where n = nb lines, m = nb columns
    :param start: grid
    :param end: grid
    :param step:
    :return:
    """
    X, Y = np.mgrid[start:end:step, start:end:step]
    Z = np.exp(-(X ** 2 + Y ** 2) / 10 - 0.5)
    n, m = np.shape(X)
    ndata = n * m
    Xi = X.reshape(1, ndata)
    Yi = Y.reshape(1, ndata)
    targets = Z.reshape(ndata)

    patterns = np.concatenate((Xi, Yi))

    patterns = np.concatenate((patterns, np.ones((1, ndata))))
    return X, Y, patterns, targets


def split_patterns(patterns, targets, prop):
    """
    randomly creates training, validation and test sets
    :param patterns:
    :param targets:
    :param prop: (train, validation, test) proportions for original set
    :return: (train, validation, test), each containing shuffled patterns and targets
    """
    n = len(patterns[0])
    n_train = int(prop[0] * n)
    n_validation = int((prop[0] + prop[1]) * n) - n_train
    sample = np.random.choice(n, int((prop[0] + prop[1]) * n), replace=False)

    if prop[2] == 0.0:
        test_set = None
        test_targets = None
    else:
        mask = np.zeros(n, dtype=bool)
        mask[sample] = True
        test_set = patterns[:, ~mask]
        test_targets = targets[~mask]

    train_set = np.zeros((3, n_train))
    train_targets = np.zeros(n_train)
    validation_set = np.zeros((3, n_validation))
    validation_targets = np.zeros(n_validation)

    for i in range(n_train):
        train_set[:, i] = patterns[:, sample[i]]
        train_targets[i] = targets[sample[i]]

    for i in range(n_train, len(sample)):
        validation_set[:, i - n_train] = patterns[:, sample[i]]
        validation_targets[i - n_train] = targets[sample[i]]

    return (train_set, train_targets), (validation_set, validation_targets), (test_set, test_targets)


def post_process(X, Y, y_out, n, m):
    X = X.reshape(n, m)
    Y = Y.reshape(n, m)
    Z = y_out.reshape(n, m)
    return X, Y, Z


def train(patterns, targets, nb_epochs, l_rate, nb_hidden_nodes,
          nb_input_nodes=2, nb_output_nodes=1, validation=None, X=None, Y=None, animate=False):
    # Init the weights
    v = np.random.normal(0, 1, (nb_hidden_nodes, nb_input_nodes + 1))
    w = np.random.normal(0, 1, (nb_output_nodes, nb_hidden_nodes + 1))
    # Store the transposed patterns
    t_patterns = np.transpose(patterns)
    # First deltas are 0
    dv, dw = 0, 0
    # Display
    if animate is True:
        n, m = np.shape(X)
        fig = plt.figure()
        plt.show(block=False)

    # Validation
    if validation is not None:
        validation_mse = np.zeros(nb_epochs)
        validation_patterns = validation[0]
        validation_targets = validation[1]
    # Epochs recorders
    eval_mse = np.zeros(nb_epochs)
    # Loop over the epochs
    for epoch in range(nb_epochs):
        # Forward pass
        y_out, h_out = predict(patterns, v, w)

        eval_mse[epoch] = mse(y_out, targets)

        # Backward pass
        delta_y = (y_out - targets) * (((1 + y_out) * (1 - y_out)) * 0.5)
        delta_h = np.dot(np.transpose(w), delta_y) * (((1 + h_out) * (1 - h_out)) * 0.5)
        delta_h = delta_h[:-1]  # Remove the bias from the forward pass

        # Weight update
        alpha = 0.9  # Momentum factor
        dv = (dv * alpha) + (1 - alpha) * np.dot(delta_h, t_patterns)
        dw = (dw * alpha) + (1 - alpha) * np.dot(delta_y, np.transpose(h_out))
        v = v - l_rate * dv
        w = w - l_rate * dw

        if validation is not None:
            y_out_v, _ = predict(validation_patterns, v, w)
            validation_mse[epoch] = mse(y_out_v, validation_targets)

        if animate is True:
            if epoch % 10 == 0 or epoch == nb_epochs - 1:
                # y_out_initial, _ = predict(initial_patterns, v, w)
                # Z = y_out_initial.reshape(n, m)
                Z = y_out.reshape(n, m)
                plot_function(fig, X, Y, Z, "Approximation at epoch " + str(epoch))
                plt.pause(.00001)

    if validation is not None:
        return v, w, eval_mse, validation_mse
    return v, w, eval_mse


def main():
    nb_hidden_nodes = 20  # Number of nodes in the hidden layer
    nb_input_nodes = 2  # Number of input nodes
    nb_output_nodes = 1  # Number of output nodes
    l_rate = 0.01  # Learning rate
    nb_epochs = 2000  # Number of training iterations

    stats_global_err = np.zeros(50)
    stats_train_err = np.zeros(50)
    stats_valid_err = np.zeros(50)
    for i in range(50):
        X, Y, patterns, targets = generate_data(-10, 10, 1)
        train_set, validation_set, test_set = split_patterns(patterns, targets, (0.8, 0.2, 0.0))
        train_patterns, train_targets = train_set
        #test_patterns, test_targets = test_set

        v, w, eval_mse, val_mse = train(train_patterns, train_targets, nb_epochs, l_rate, nb_hidden_nodes,
                               validation=validation_set, X=X, Y=Y, animate=False)
        stats_train_err[i] = eval_mse[-1]
        stats_valid_err[i] = val_mse[-1]
        y_out, _ = predict(patterns, v, w)
        stats_global_err[i] = mse(y_out, targets)
        #print("final error", mse(y_out, targets), eval_mse[len(eval_mse) - 1], val_mse[len(val_mse) - 1])
        #fig = plt.figure()
        #Z = y_out.reshape(np.shape(X))
        #plot_function(fig, X, Y, Z, "final")
        #plt.show()

        #plt.clf()

        #support = np.arange(0, nb_epochs, 1)
        #plt.plot(support, eval_mse, val_mse)
        #plt.legend(['train', 'validation'])
        #plt.show()
    print("mean train, val, global", np.mean(stats_train_err), np.mean(stats_valid_err), np.mean(stats_global_err))
    print("std train, val, global", np.std(stats_train_err), np.std(stats_valid_err), np.std(stats_global_err))
    """
    nodes_mse = np.zeros(25)
    for i in range(1, 26):
        v, w, eval_mse = train(patterns, targets, nb_epochs, l_rate, i, validation=None, X=X, Y=Y, animate=False)
        # Compute the error
        y_out, _ = predict(patterns, v, w)
        error = mse(y_out, targets)
        #print("Final mse: " + str(error))
        nodes_mse[i - 1] = error
    """

    #plt.plot(np.arange(1, 26), nodes_mse)
    #plt.show()



if __name__ == "__main__":
    main()
