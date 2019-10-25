from datagen import generate_data, plot_classes, lab_generate_data
from drawnow import drawnow
import numpy as np
import matplotlib.pyplot as plt
from time import time


def process_inputs(classA, classB, bias=True):
    """
    given classA and classB, creates matrix X of shuffled inputs with a third row for biases and a fourth row for
    remembering which class the point belonged to (used to create target vector)
    :param classA:
    :param classB:
    :return:
    """
    _, n = np.shape(classA)
    # Adding the bias
    if bias is True:
        classA = np.concatenate((classA, np.ones((1, n))))
        classB = np.concatenate((classB, np.ones((1, n))))

    # Remembering the labels
    classA = np.concatenate((classA, np.ones((1, n))))
    classB = np.concatenate((classB, -np.ones((1, n))))

    X = np.concatenate((classA, classB), axis=1)
    X = np.transpose(X)
    np.random.shuffle(X)
    X = np.transpose(X)
    return X


def compute_mse(W, X, bias):
    """
    mean squared error
    :param W:
    :param X:
    :return:
    """
    if bias is True:
        T = X[3,:]
        results = np.sign(np.dot(W, X[0:3, :]))
    else:
        T = X[2,:]
        results = np.sign(np.dot(W, X[0:2, :]))
    error = 0

    for i in range(len(results)):
        error += (T[i] - results[i])**2
    return error / len(results)


def batch_train(eta, X, epochs=None, nb_batches=1, bias=True, classA=None, classB=None):
    """
    trains the weights
    :param epochs:
    :param eta: learning rate parameter
    :param X:
    :return: trained W
    """
    if bias is True:
        W = np.random.normal(0, 1, 3)
        T = X[3,:]
    else:
        W = np.random.normal(0, 1, 2)
        T = X[2,:]
    _, n = np.shape(X)

    def drawfig():
        plot_classes(classA, classB, W)

    if epochs is not None:
        accuracy = np.zeros((epochs))
        error = np.zeros((epochs))
        for i in range(epochs):
            X = np.transpose(X)
            np.random.shuffle(X)
            X = np.transpose(X)
            for j in range(nb_batches):
                start = int(j/nb_batches * n)
                end = int((j+1)/nb_batches * n) - 1
                W = update_W(T[start:end], W, X[:, start:end], eta, bias)
                #drawnow(drawfig)
            accuracy[i] = compute_accuracy(W, X, bias)
            error[i] = compute_mse(W, X, bias)
    else:
        accuracy = []
        error = []
        while compute_accuracy(W, X) < 1.0:
            W = update_W(T, W, X, eta)
            accuracy.append(compute_accuracy(W, X))
            error.append(compute_mse(W, X))

    return W, accuracy, error


def seq_update_W(T, W, X, eta):
    _, n = np.shape(X)
    for i in range(n):
        W = W - eta * np.dot((np.dot(W, X[0:3, i]) - T[i]), np.transpose(X[0:3, i]))
    return W


def perceptron_update_W(T, W, X, eta):
    _, n = np.shape(X)
    for i in range(n):
        W = W - eta * np.dot((np.sign(np.dot(W, X[0:3, i])) - T[i]), X[0:3, i])
    return W


def perceptron_train(eta, X, epochs):
    T = X[3, :]
    W = np.random.normal(0, 1, 3)
    accuracy = np.zeros((epochs))
    error = np.zeros((epochs))
    for i in range(epochs):
        W = perceptron_update_W(T, W, X, eta)
        accuracy[i] = compute_accuracy(W, X, True)
        error[i] = compute_mse(W, X, True)
        X = np.transpose(X)
        np.random.shuffle(X)
        X = np.transpose(X)
    return W, accuracy, error


def seq_train(eta, X, epochs):
    T = X[3, :]
    W = np.random.normal(0, 1, 3)
    accuracy = np.zeros((epochs))
    error = np.zeros((epochs))
    for i in range(epochs):
        W = seq_update_W(T, W, X, eta)
        accuracy[i] = compute_accuracy(W, X, True)
        error[i] = compute_mse(W, X, True)
        X = np.transpose(X)
        np.random.shuffle(X)
        X = np.transpose(X)
    return W, accuracy, error


def update_W(T, W, X, eta, bias):
    delta_W = delta_rule(eta, W, X, T, bias)
    W = W + delta_W
    return W


def delta_rule(eta, W, X, T, bias):
    """
    applies delta rule
    :param eta:
    :param W:
    :param X:
    :param T:
    :return:
    """

    if bias is True:
        return - eta * np.dot((np.dot(W, X[0:3, :]) - T), np.transpose(X[0:3, :]))
    else:
        return - eta * np.dot((np.dot(W, X[0:2, :]) - T), np.transpose(X[0:2, :]))


def compute_accuracy(W, X, bias):
    """
    computes as a percentage how many inputs were correctly classified
    :param W:
    :param X:
    :return:
    """

    if bias is True:
        T = X[3,:]
        results = np.sign(np.dot(W, X[0:3, :]))
    else:
        T = X[2,:]
        results = np.sign(np.dot(W, X[0:2, :]))
    accuracy = 0
    for i in range(len(results)):
        if T[i] == results[i]:
            accuracy += 1
    return accuracy / len(results)


def plot_accuracy(accuracy, epochs):
    x = np.arange(1, epochs + 1, 1)
    plt.plot(x, accuracy)
    plt.title('Accuracy of classification depending on number of epochs')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.show()


def plot_error(mse, epochs):
    x = np.arange(1, epochs + 1, 1)
    plt.plot(x, mse)
    plt.title('Mean squared error depending on number of epochs')
    plt.xlabel('Mean squared error')
    plt.ylabel('Accuracy')
    plt.show()


def batch_train_display():
    epochs = 100
    classA, classB = generate_data(100)
    X = process_inputs(classA, classB)
    W, accuracy, error = batch_train(0.001, X, epochs)
    plot_classes(classA, classB, W)
    plot_accuracy(accuracy, len(accuracy))
    #plot_error(error, len(accuracy))
    print("final accuracy is", accuracy[len(accuracy) - 1])
    print("number of epochs:", len(accuracy))


def seq_train_display():
    """
    sequential training using delta rule
    :return:
    """
    epochs = 50
    classA, classB = generate_data(100)
    X = process_inputs(classA, classB)
    W, accuracy, error = seq_train(0.01, X, epochs)
    plot_classes(classA, classB, W)
    plot_accuracy(accuracy, len(accuracy))
    plot_error(error, len(accuracy))
    print("final accuracy is", accuracy[len(accuracy) - 1])
    print("number of epochs:", len(accuracy))


def perceptron_train_display():
    """
    sequential training using delta rule
    :return:
    """
    epochs = 50
    classA, classB = generate_data(100)
    X = process_inputs(classA, classB)
    W, accuracy, error = perceptron_train(0.006, X, epochs)
    plot_classes(classA, classB, W)
    plot_accuracy(accuracy, len(accuracy))
    plot_error(error, len(accuracy))
    print("final accuracy is", accuracy[len(accuracy) - 1])
    print("number of epochs:", len(accuracy))


def adjust_eta(train_function):
    """
    This functions shows how training is the most efficient when learning rate eta = 0.002
    :param train_function: sequential or batch learning
    :return:
    """
    epochs = 50
    classA, classB = generate_data(100)
    X = process_inputs(classA, classB)
    eta = np.arange(0.001, 0.01, 0.0005)
    stats = []
    for k in range(len(eta)):
        acc = []
        for i in range(50):
            W, accuracy, error = train_function(eta[k], X, epochs)
            acc.append(accuracy[epochs-1])
        stats.append(np.mean(acc))
    plt.plot(eta, stats)
    plt.xlabel('learning rate (eta)')
    plt.ylabel('accuracy')
    plt.title('accuracy depending on learning rate')
    #plt.show()


def adjust_epochs(train_function):
    """
    This functions shows how training is the most efficient when epochs >= 40 for small learning rates
    :return:
    """
    epochs = np.arange(1, 10, 1)
    classA, classB = generate_data(100)
    X = process_inputs(classA, classB)
    eta = 0.01
    stats = []
    for k in range(len(epochs)):
        acc = []
        for i in range(100):
            W, accuracy, error = train_function(eta, X, epochs[k])
            acc.append(accuracy[len(accuracy)-1])
        stats.append(np.mean(acc))
    plt.plot(epochs, stats)
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.title('accuracy depending on number of epochs')
    plt.show()
    print(epochs, acc)


def adjust_batch():
    epochs = 50
    nb_batches = np.arange(1, 11, 1)
    classA, classB = generate_data(100)
    X = process_inputs(classA, classB)
    eta = 0.01
    stats = []
    for k in range(len(nb_batches)):
        acc = []
        for i in range(50):
            W, accuracy, error = batch_train(eta, X, epochs, nb_batches[k])
            acc.append(accuracy[len(accuracy)-1])
        stats.append(np.mean(acc))
    plt.plot(nb_batches, stats)
    plt.xlabel('number of batches')
    plt.ylabel('accuracy')
    plt.title('accuracy depending on number of batches')
    plt.show()


def sensibility():
    epochs = 100
    np.random.seed(0)
    classA, classB = generate_data(100)
    np.random.seed(4)  # seed 1 and 4
    X = process_inputs(classA, classB)
    W, accuracy, error = batch_train(0.001, X, epochs)
    plot_classes(classA, classB, W)
    plot_accuracy(accuracy, len(accuracy))
    print("final accuracy is", accuracy[len(accuracy) - 1])
    print("number of epochs:", len(accuracy))


def no_bias_batch_train():
    epochs = 200
    np.random.seed(0)
    classA, classB = generate_data(100, mA = np.array([2.0, 4]), mB=np.array([-5, -2]), sigmaA=1.5, sigmaB=1.5)
    #classA, classB = generate_data(100)
    np.random.seed(1)  # seed 1 and 2
    X = process_inputs(classA, classB, False)
    W, accuracy, error = batch_train(0.0001, X, epochs, 1, False)
    plot_classes(classA, classB, W, False)
    plot_accuracy(accuracy, len(accuracy))
    print("final accuracy is", accuracy[len(accuracy) - 1])
    print("number of epochs:", len(accuracy))


def process_inputs_drop_AB(classA, classB):
    """
    removes 25% of A and 25% of B
    creates input matrix X and makes a shuffle
    :param classA:
    :param classB:
    :return:
    """
    _, n = np.shape(classA)
    classA, classB = add_bias_labels(classA, classB, n)

    #drop 25% of A and B
    for i in range(n//4):
        col_A = np.random.choice(len(classA[0]))
        col_B = np.random.choice(len(classB[0]))
        classA = np.delete(classA, col_A, 1)
        classB = np.delete(classB, col_B, 1)

    X = np.concatenate((classA, classB), axis=1)
    X = np.transpose(X)
    np.random.shuffle(X)
    X = np.transpose(X)
    return X


def process_inputs_drop_A(classA, classB):
    """
    removes 50% of A
    creates input matrix X and makes a shuffle
    :param classA:
    :param classB:
    :return:
    """
    _, n = np.shape(classA)
    classA, classB = add_bias_labels(classA, classB, n)

    #drop 50% of A
    for i in range(n//2):
        col_A = np.random.choice(len(classA[0]))
        classA = np.delete(classA, col_A, 1)

    X = np.concatenate((classA, classB), axis=1)
    X = np.transpose(X)
    np.random.shuffle(X)
    X = np.transpose(X)
    return X


def process_inputs_drop_B(classA, classB):
    """
    removes 50% of B
    creates input matrix X and makes a shuffle
    :param classA:
    :param classB:
    :return:
    """
    _, n = np.shape(classA)
    classA, classB = add_bias_labels(classA, classB, n)

    #drop 50% of B
    for i in range(n//2):
        col_B = np.random.choice(len(classB[0]))
        classB = np.delete(classB, col_B, 1)

    X = np.concatenate((classA, classB), axis=1)
    X = np.transpose(X)
    np.random.shuffle(X)
    X = np.transpose(X)
    return X


def process_inputs_drop_20_80(classA, classB):
    """
    removes 20% of A where x < 0 and 80% of A where x > 0
    creates input matrix X and makes a shuffle
    :param classA:
    :param classB:
    :return:
    """
    _, n = np.shape(classA)
    classA, classB = add_bias_labels(classA, classB, n)

    # separate classA in <0 and >0
    classA_minus = classA[:, (classA[1,:] < 0)]
    classA_plus = classA[:, (classA[1,:] > 0)]

    #drop 20% of classA_minus
    n_minus = len(classA_minus[0])
    for i in range(n_minus // 5):
        col = np.random.choice(len(classA_minus[0]))
        classA_minus = np.delete(classA_minus, col, 1)

    #drop 80% of classA_plus
    n_plus = len(classA_plus[0])
    for i in range(4 * n_plus // 5):
        col = np.random.choice(len(classA_plus[0]))
        classA_plus = np.delete(classA_plus, col, 1)

    classA = np.concatenate((classA_minus, classA_plus), axis=1)
    print(len(classA[0]))
    X = np.concatenate((classA, classB), axis=1)
    X = np.transpose(X)
    np.random.shuffle(X)
    X = np.transpose(X)
    return X


def add_bias_labels(classA, classB, n):
    # Adding the bias
    classA = np.concatenate((classA, np.ones((1, n))))
    classB = np.concatenate((classB, np.ones((1, n))))
    # Remembering the labels
    classA = np.concatenate((classA, np.ones((1, n))))
    classB = np.concatenate((classB, -np.ones((1, n))))
    return classA, classB


def not_l_separable():
    epochs = 500
    classA, classB = lab_generate_data(n=100, mA = np.array([5.0, 0.3]), mB = np.array([0.0, -0.1]), sigmaA = 0.2, sigmaB = 0.3)
    X = process_inputs_drop_A(classA, classB)
    W, accuracy, error = perceptron_train(0.0001, X, epochs)
    plot_classes(classA, classB, W)
    plot_accuracy(accuracy, len(accuracy))
    print("final accuracy is", accuracy[len(accuracy) - 1])
    print("number of epochs:", len(accuracy))



def main():
    batch_train_display()

if __name__ == '__main__':
    main()
