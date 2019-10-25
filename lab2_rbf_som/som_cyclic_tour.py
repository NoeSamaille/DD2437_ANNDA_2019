import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def create_weight_matrix():
    w = np.random.uniform(0, 1, (10, 2))
    return w


def find_closest_w_row(x, w):
    d = np.zeros(10)
    for i in range(10):
        d[i] = euclidean(x, w[i, :])
    return np.argmin(d)


def is_neighbour(i, ind, size):
    """
    :param i: node index to check if it is a neighbour
    :param ind: checking if i is a neighbour of this node ind.
    :param size: size of neighbourhood. Ex size = 1 means i-1 and i+1 are neighbours.
    :return: True if it is a neighbour, False otherwise
    """
    max_ind = np.max([i, ind])
    min_ind = np.min([i, ind])
    return (max_ind - min_ind <= size) or (max_ind - min_ind >= (10-size))


def update_w(x, w, ind, eta, eta_n, size):
    """
    updates winner and its neighbours
    :param w:
    :param ind:
    :return:
    """
    for i in range(10):
        if i == ind:
            w[i, :] += eta * (x - w[i, :])
        else:
            w[i, :] += eta_n * is_neighbour(i, ind, size) * (x - w[i, :])
    return w


def move_neurons(eta, eta_n, cities, size, w):
    for i in range(10):
        ind = find_closest_w_row(cities[i, :], w)
        w = update_w(cities[i, :], w, ind, eta, eta_n, size)


def train(cities, eta=0.2, nb_epochs=20):
    alpha = 0.5
    eta_n = eta / 2
    w = create_weight_matrix()
    neighbourhood_size = 2
    for i in range(nb_epochs):
        print("Epoch", i)
        print("->", eta)
        print("->", eta_n)
        print("->", neighbourhood_size)
        move_neurons(eta, eta_n, cities, neighbourhood_size, w)
        # reduce learning rate and neighbourhood
        eta = alpha * eta**(i / nb_epochs)
        eta_n = alpha * eta_n**(i / nb_epochs)
        neighbourhood_size = round(2*((nb_epochs-i)/nb_epochs))
    return w


def order_cities(cities, w):
    pos = np.zeros(10)
    for i in range(10):
        ind = find_closest_w_row(cities[i, :], w)
        pos[i] = ind
    order = np.argsort(pos)
    order = np.append(order, order[0])
    print(order)
    plt.title("Cyclic tour")
    plt.scatter(cities[order][:, 0], cities[order][:, 1])
    plt.plot(cities[order][:, 0], cities[order][:, 1])
    plt.show()


def import_cities():
    return np.loadtxt('data/cities.dat', comments="%", delimiter=',').reshape((10, 2))


def main():
    cities = import_cities()
    w = train(cities)
    order_cities(cities, w)


if __name__ == "__main__":
    main()