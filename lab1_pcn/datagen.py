"""
Assignement Part I
Generation of linearly-separable data
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(n, mA = np.array([2.0, 1.5]), mB = np.array([0.0, 0.5]), sigmaA = 0.4, sigmaB = 0.2):
    """
    Generates correlated 2D points from multivariate normal distribution. There are two classes A and B with different
    distribution.
    :param n: number of points for each class
    :return: classA and classB points
    """

    classA = np.zeros((2, n))
    classB = np.zeros((2, n))

    for i in range(2):
        classA[i,:] = np.random.normal(mA[i], sigmaA, n)
        classB[i,:] = np.random.normal(mB[i], sigmaB, n)

    return classA, classB


def lab_generate_data(n=100, mA = np.array([1.0, 0.3]), mB = np.array([0.0, -0.1]), sigmaA = 0.2, sigmaB = 0.3):
    """
    Generates correlated 2D points from multivariate normal distribution. There are two classes A and B with different
    distribution.
    :param n: number of points for each class
    :return: classA and classB points
    """

    classA = np.zeros((2, n))
    classB = np.zeros((2, n))

    for i in range(2):
        if i == 0:
            classA[i,0:n//2] = np.random.normal(-mA[i], sigmaA, n//2)
            classA[i, n//2:] = np.random.normal(mA[i], sigmaA, n//2)
        else:
            classA[i, :] = np.random.normal(mA[i], sigmaA, n)
        classB[i,:] = np.random.normal(mB[i], sigmaB, n)

    return classA, classB


def plot_classes(classA, classB, W = None, bias = True):
    """
    Plots points from A and B classes
    :param classA:
    :param classB:
    :param W: (w0, w1, w2)
    :return:
    """
    ax = plt.subplot(111)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    plt.axis('equal')
    plt.title('Uneven class representation but representative sample distribution')
    plt.scatter(classA[0,:], classA[1,:], color='red')
    plt.scatter(classB[0,:], classB[1,:], color='blue')
    plt.legend(['A', 'B'])

    if W is not None:
        x = np.arange(-2, 2, 1)
        if bias is True:
            w0, w1, w2 = W[2], W[0], W[1]
            plt.plot(x, -1/w2 * (w0 + w1 * x))
            plt.quiver(-w0 / np.linalg.norm(W), -w0 / np.linalg.norm(W), w1, w2, color=['g'], scale=10)
        else:
            w1, w2 = W[0], W[1]
            plt.plot(x, -1 / w2 * w1 * x)
            plt.quiver(0, 0, w1, w2, color=['g'], scale=10)

    plt.show()


def main():
    W = np.array([1, 1, 1])
    classA, classB = generate_data(100)
    plot_classes(classA, classB, W)


if __name__ == '__main__':
    main()
