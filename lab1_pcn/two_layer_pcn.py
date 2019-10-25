import math
import numpy as np
import matplotlib.pyplot as plt
from datagen import generate_data
from datagen import lab_generate_data
from datagen import plot_classes
from delta_rule import process_inputs


def predict(patterns, v, w):
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


def seq_predict(patterns, v, w):
    # Forward pass threw the network
    # Compute hidden layer input
    h_in = np.dot(v, patterns)
    # Compute hidden layer output
    h_out = (2 / (1 + np.exp(-1 * h_in))) - 1
    # Add the bias for the hidden layer
    h_out = np.concatenate((h_out, np.ones((1, len(h_out[0])))))
    # Compute output layer input
    y_in = np.dot(w, h_out)
    # Compute output layer output (= NN output)
    y_out = (2 / (1 + np.exp(-1 * y_in))) - 1
    return y_out, h_out


def accuracy(patterns, targets, v, w):
    y_out, _ = predict(patterns, v, w)
    e = targets - np.sign(y_out)
    accuracy = len(e[(e[:] == 0)])
    return accuracy / len(targets)


def mse(patterns, targets, v, w):
    y_out, _ = predict(patterns, v, w)
    error = targets - np.sign(y_out)
    return np.sum(error**2) / len(targets)


def train(patterns, targets, nb_epochs, l_rate, nb_hidden_nodes, nb_input_nodes=2, nb_output_nodes=1, nb_batches=1, init_weights_sd=1):
    # Init the weights
    v = np.random.normal(0, init_weights_sd, (nb_hidden_nodes, nb_input_nodes + 1))
    w = np.random.normal(0, init_weights_sd, (nb_output_nodes, nb_hidden_nodes + 1))
    # Store the transposed patterns
    t_patterns = np.transpose(patterns)
    # First deltas are 0
    dv, dw = 0, 0
    # Loop over the epochs
    for epoch in range(nb_epochs):
        # for batch in range(nb_batches):
        #print("Epoch " + str(epoch) + " mse: " + str(mse(patterns, targets, v, w)))
        # Forward pass
        y_out, h_out = predict(patterns, v, w)
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
    return v, w


def seq_train(patterns, targets, nb_epochs, l_rate, nb_hidden_nodes, nb_input_nodes=2, nb_output_nodes=1, nb_batches=1, init_weights_sd=1):
    # Init the weights
    v = np.random.normal(0, init_weights_sd, (nb_hidden_nodes, nb_input_nodes + 1))
    w = np.random.normal(0, init_weights_sd, (nb_output_nodes, nb_hidden_nodes + 1))
    # Store the transposed patterns
    t_patterns = np.transpose(patterns)
    # First deltas are 0
    dv, dw = 0, 0
    # Loop over the epochs
    for epoch in range(nb_epochs):
        # for batch in range(nb_batches):
        #print("Epoch " + str(epoch) + " mse: " + str(mse(patterns, targets, v, w)))
        # Forward pass
        for i in range(len(patterns[0])):
            y_out, h_out = seq_predict(patterns[:, i].reshape(3, 1), v, w)
            # Backward pass
            delta_y = (y_out - targets[0, i]) * (((1 + y_out) * (1 - y_out)) * 0.5)
            delta_h = np.dot(np.transpose(w), delta_y) * (((1 + h_out) * (1 - h_out)) * 0.5)
            delta_h = delta_h[:-1] # Remove the bias from the forward pass
            # Weight update
            alpha = 0.9 # Momentum factor
            dv = (dv * alpha) + (1 - alpha) * np.dot(delta_h, t_patterns[i, 0])
            dw = (dw * alpha) + (1 - alpha) * np.dot(delta_y, np.transpose(h_out))
            v = v - l_rate * dv
            w = w - l_rate * dw
    return v, w


def plot_boundaries(classA, classB, v):
    ax = plt.subplot(111)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    plt.scatter(classA[0,:], classA[1,:], label='Class A')
    plt.scatter(classB[0,:], classB[1,:], label='Class B')
    # Loop over the hidden nodes
    for i in range(len(v)):
        v1, v2, theta = v[i][0], v[i][1], -1 * v[i][2]
        x = np.arange(-2, 3, 1)
        plt.ylim(-5, 5)
        plt.plot(x, 1/v2 * (-v1*x + theta)) # x2 = (-x1*v1 + theta) / v2
    plt.title('Classification decision boundaries')
    plt.show()


def auto_encoder():
    nb_input_nodes = 8  # Number of input nodes
    nb_hidden_nodes = 3  # Number of nodes in the hidden layer
    nb_output_nodes = 8  # Number of output nodes
    l_rate = 0.1  # Learning rate
    nb_epochs = 1000  # Number of training iterations

    # Init the input patterns and targets
    patterns = [[1, -1, -1, -1, -1, -1, -1, -1],
                [-1, 1, -1, -1, -1, -1, -1, -1],
                [-1, -1, 1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, 1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 1, -1],
                [-1, -1, -1, -1, -1, -1, -1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]
    #patterns = np.transpose(patterns)
    #np.random.shuffle(patterns)
    #patterns = np.transpose(patterns)
    targets = patterns[:-1]

    # Train the NN
    v, w = train(patterns, targets, nb_epochs, l_rate, nb_hidden_nodes, nb_input_nodes, nb_output_nodes, init_weights_sd=0.1)

    # Get the encoding (h_out)
    _, h_out = predict(patterns, v, w)
    print(np.transpose(np.sign(h_out[:-1])))

    # Assess NN performance
    err = mse(patterns, targets, v, w)
    print("Final mse: " + str(err))
    return err


def binary_classification():
    nb_data = 100  # Number of input patterns
    nb_hidden_nodes = 2  # Number of nodes in the hidden layer
    nb_input_nodes = 2  # Number of input nodes
    nb_output_nodes = 1  # Number of output nodes
    l_rate = 0.001  # Learning rate
    nb_epochs = 1000  # Number of training iterations

    # Init the input patterns
    classA, classB = lab_generate_data(int(nb_data/2))
    # Add bias, targets and shuffle the inputs
    inputs = process_inputs(classA, classB)
    # Split inputs into patterns and targets
    patterns, targets = np.split(inputs, [3])
    # Train the NN
    v, w = seq_train(patterns, targets, nb_epochs, l_rate, nb_hidden_nodes)
    # Compute the accuracy
    print("Final mse: " + str(mse(patterns, targets, v, w)))
    plot_boundaries(classA, classB, v)
    print("Final acc: " + str(accuracy(patterns, targets, v, w)))


def encoder_convergence_rate():
    win = 0
    for i in range(100):
        if auto_encoder() == 0.0:
            win += 1
    return win/100


def main():
    #auto_encoder()
    binary_classification()


if __name__ == "__main__":
    main()