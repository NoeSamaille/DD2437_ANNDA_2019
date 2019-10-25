import numpy as np
import matplotlib.pyplot as plt
from datagen import generate_data


def predict(input_nodes, weights, neuron):
    """
    Compute the activation of the neuron neuron for a single input vector.

    Params:
        input_nodes: input vector
        weights: weights matrix
        neuron: neuron index

    Returns:
        1.0 if neuron fires for the input vector,
        0.0 otherwise
    """
    activation = 0.0
    # Loop over the input nodes
    for i in range(len(input_nodes)):
        activation +=  input_nodes[i] * weights[i][neuron]
    return 1.0 if activation > 0.0 else 0.0


def accuracy(input_vectors, targets, weights):
    """
    Compute the accuracy of the network.

    Params:
        input_vectors: set of input vectors
        targets: target labels
        weights: weights matrix

    Returns:
        float ratio: Number of correct classification / Total number of classification
    """
    nb_success = 0.0
    # Loop over the input vectors
    for d in range(np.shape(input_vectors)[0]):
        # Loop over the neurons
        for j in range(np.shape(weights)[1]):
            prediction = predict(input_vectors[d], weights, j)
            if prediction == targets[d][j]:
                nb_success += 1.0
    # Return number of success / total
    return nb_success/(np.shape(input_vectors)[0]*np.shape(weights)[1])


def plot(classA, classB, weights=[]):
    """
    Plot the clouds of points for the classA and classB.
    If weights is set, plot the network decision boundary.

    Params:
        input_vectors: set of input vectors
        targets: target labels
        weights: weights matrix

    Returns:
        float ratio: Number of correct classification / Total number of classification
    """
    ax = plt.subplot(111)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    plt.scatter(classA[0,:], classA[1,:], label='class A')
    plt.scatter(classB[0,:], classB[1,:], label='class B')
    if len(weights) > 0:
        # Plot the decision boundary
        w0, w1, w2 = weights[0][0], weights[1][0], weights[2][0]
        x = np.arange(-2, 3, 1)
        plt.ylim(-5, 5)
        plt.plot(x, 1/w2 * (-w1*x - w0), label='Decision boundary') # x2 = (-x1*w1 + theta) / w2
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Decision boundary of the 2 classes A and B')
    plt.legend(loc='upper right')
    plt.show()


def seq_train(input_vectors, targets, weights, nb_epochs, l_rate, verbose=False, do_plot=False, classA=None, classB=None):
    """
    Perform the sequential training of the perceptron

    Params:
        input_vectors: set of input vectors
        targets: target labels
        weights: weights matrix
        nb_epochs: max number of epochs
        l_rate: learning rate
    
    Returns:
        Resulting weights matrix
    """
    for epoch in range(nb_epochs):
        # Compute current weights accuracy
        acc = accuracy(input_vectors, targets, weights)
        if verbose:
            print("Epoch no: ", epoch)
            print("Accuracy: ", acc)
        if do_plot and len(classA) and len(classB):
            plot(classA, classB, weights=weights)
        if acc == 1.0:
            break
        # Loop over the input vectors
        for d in range(np.shape(input_vectors)[0]):
            # Loop over the neurons
            for j in range(np.shape(weights)[1]):
                # Compute the activation of the neuron j for input vector d
                activation = predict(input_vectors[d], weights, j)
                # Update weights
                if activation - targets[d][j] != 0.0:
                    # Loop over the input nodes
                    for i in range(np.shape(weights)[0]):
                        weights[i][j] -= l_rate*(activation - targets[d][j])*input_vectors[d][i]
    return weights, epoch


def batch_train(input_vectors, targets, weights, nb_epochs, l_rate):
    """
    Perform the batch training of the perceptron

    Params:
        input_vectors: set of input vectors
        targets: target labels
        weights: weights matrix
        nb_epochs: max number of epochs
        l_rate: learning rate
    
    Returns:
        Resulting weights matrix
    """
    for epoch in range(nb_epochs):
        # Compute current weights accuracy
        acc = accuracy(input_vectors, targets, weights)
        print("Epoch no: ", epoch)
        print("Accuracy: ", acc)
        if acc == 1.0:
            break
        # Adder
        activations = np.dot(input_vectors, weights)
        # Thresholder
        activations = np.where(activations > 0, 1, 0)
        # Weights update
        weights -= l_rate*np.dot(np.transpose(input_vectors), activations - targets)
    return weights


def seq_pcn_stats():
    """
    Shows the efficiency of the sequential perceptron when eta varies
    """
    n_data = 200 # Number input vectors
    m = 2 # Number of features (input nodes)
    n = 1 # Number of output nodes
    nb_epochs = 10 # Max number of iterations

    # Generate input data
    classA, classB = generate_data(int(n_data/2))
    input_vectors = np.concatenate((classA, classB), axis=1)
    # Adding the bias
    input_vectors = np.concatenate((np.ones((1, n_data)), input_vectors))
    # Transposing matrix
    input_vectors = np.transpose(input_vectors) # Matrix n_data by m+1
    # Targets
    targets = np.zeros((n_data, n))
    for i in range(int(n_data/2)):
        for j in range(n):
            targets[i][j] = 1 # Neurons should fire 1 for classA points

    eta = np.arange(0.001, 0.011, 0.0001)
    stats_epoch = []
    stats_accuracy = []
    for k in range(len(eta)):
        epochs = []
        accuracies = []
        for i in range(20):
            # Init the weights
            weights = np.random.normal(0, 1, (m + 1, n))
            weights, epoch = seq_train(input_vectors, targets, weights, nb_epochs, eta[k])
            acc = accuracy(input_vectors, targets, weights)
            epochs.append(epoch)
            accuracies.append(acc)
        stats_epoch.append(np.mean(epochs))
        stats_accuracy.append(np.mean(accuracies))
    plt.plot(eta, stats_epoch)
    plt.xlabel('Learning rate (eta)')
    plt.ylabel('Epochs')
    plt.title('Required number of epochs depending on learning rate')
    plt.show()
    plt.plot(eta, stats_accuracy)
    plt.xlabel('Learning rate (eta)')
    plt.ylabel('Accuracy')
    plt.title('NN Accuracy depending on learning rate')
    plt.show()


def plot_boundaries():
    """
    Plots the decision boundary updates while training
    """
    n_data = 200 # Number input vectors
    m = 2 # Number of features (input nodes)
    n = 1 # Number of output nodes
    l_rate = 0.006 # Learning rate
    nb_epochs = 20 # Max number of iterations

    # Generate input data
    classA, classB = generate_data(int(n_data/2))
    input_vectors = np.concatenate((classA, classB), axis=1)
    # Adding the bias
    input_vectors = np.concatenate((np.ones((1, n_data)), input_vectors))
    # Transposing matrix
    input_vectors = np.transpose(input_vectors) # Matrix n_data by m+1
    # Targets
    targets = np.zeros((n_data, n))
    for i in range(int(n_data/2)):
        for j in range(n):
            targets[i][j] = 1 # Neurons should fire 1 for classA points
    # Init the weights
    weights = np.random.rand(m + 1, n) * 0.1 - 0.05
    weights = seq_train(input_vectors, targets, weights, nb_epochs, l_rate, verbose=True, do_plot=True, classA=classA, classB=classB)


def main():
    plot_boundaries()
    #seq_pcn_stats()


if __name__ == '__main__':
    main()
