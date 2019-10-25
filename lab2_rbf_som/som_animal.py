import numpy as np
from scipy.spatial.distance import euclidean


def import_attributes():
    props = np.loadtxt('data/animals.dat', dtype='i', delimiter=',')
    props = props.reshape((32, 84))
    return props


def create_weight_matrix():
    w = np.random.uniform(0, 1, (100, 84))
    return w


def is_neighbour(i, ind, size):
    """
    :param i: node index to check if it is a neighbour
    :param ind: checking if i is a neighbour of this node ind.
    :param size: size of neighbourhood. Ex size = 1 means i-1 and i+1 are neighbours.
    :return: True if it is a neighbour, False otherwise
    """
    return (i >= ind - size) and (i <= ind + size)


def update_w(x, w, ind, eta, eta_n, size):
    """
    updates winner and its neighbours
    :param w:
    :param ind:
    :return:
    """
    for i in range(32):
        if i == ind:
            w[i, :] += eta * (x - w[i, :])
        else:
            w[i, :] += eta_n * is_neighbour(i, ind, size) * (x - w[i, :])
    return w


def find_closest_w_row(x, w):
    d = np.zeros(100)
    for i in range(100):
        d[i] = euclidean(x, w[i, :])
    return np.argmin(d)


def train(props, eta=0.2, nb_epochs=20):
    alpha = 0.4
    eta_n = eta / 2
    w = create_weight_matrix()
    neighbourhood_size = 50
    for i in range(nb_epochs):
        move_neurons(eta, eta_n, props, neighbourhood_size, w)

        # reduce learning rate and neighbourhood
        eta = alpha * eta**(i / nb_epochs)
        eta_n = alpha * eta_n**(i / nb_epochs)
        neighbourhood_size = alpha * neighbourhood_size**(i / nb_epochs)
    return w


def import_animals():
    animals = np.loadtxt('data/animalnames.txt', delimiter='\n', dtype='<U11')
    return animals


def order_animals(props, w):
    animals = import_animals()
    pos = np.zeros(32)
    for i in range(32):
        ind = find_closest_w_row(props[i, :], w)
        pos[i] = ind
    sort_index = np.argsort(pos)
    print(animals[sort_index])
    # for ind in range(32):
    #     print(animals[sort_index[ind]], sort_index[ind])


def move_neurons(eta, eta_n, props, size, w):
    for i in range(32):
        ind = find_closest_w_row(props[i, :], w)
        w = update_w(props[i, :], w, ind, eta, eta_n, size)


if __name__ == "__main__":
    props = import_attributes()
    w = train(props)
    order_animals(props, w)
