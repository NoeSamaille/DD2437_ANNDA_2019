import numpy as np
from scipy.spatial.distance import euclidean
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import svgwrite


def import_votes():
    print(os.getcwd())
    props = np.loadtxt('data/votes.dat', dtype='i', delimiter=',')
    return props.reshape((349, 31))


def create_weight_matrix():
    w = np.random.uniform(0, 1, (100, 31))
    return w


def is_neighbour(line, col, ind, size):
    """
    :param i: node index to check if it is a neighbour
    :param ind: checking if i is a neighbour of this node ind.
    :param size: size of neighbourhood. Ex size = 1 means i-1 and i+1 are neighbours.
    :return: True if it is a neighbour, False otherwise
    """
    ind_line = ind // 10
    ind_col = ind % 10
    return (line >= ind_line - size) and (line <= ind_line + size) \
           and (col >= ind_col - size) and (col <= ind_col + size)


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
            line = ind // 10
            col = ind % 10
            w[i, :] += eta_n * is_neighbour(line, col, ind, size) * (x - w[i, :])
    return w


def find_closest_w_row(x, w):
    d = np.zeros(100)
    for i in range(100):
        d[i] = euclidean(x, w[i, :])
    return np.argmin(d)


def train(props, eta=0.2, nb_epochs=10):
    alpha = 0.5
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


def import_mps():
    mps = np.loadtxt('data/mpnames.txt', delimiter='\n', dtype='U')
    return mps


def sort_mps(mps, props, w):
    pos = np.zeros(349, dtype=int)
    for i in range(349):
        ind = find_closest_w_row(props[i, :], w)
        pos[i] = ind
    sort_index = np.argsort(pos)
    sorted_mps = mps[sort_index]
    pos.sort()
    return sorted_mps, pos


def gender_map(sorted_mps, pos, d):
    gender_map = [[] for _ in range(100)]
    for i in range(349):
        gender_map[pos[i]].append(d[sorted_mps[i]][0])
    for j in range(100):
        if gender_map[j]:
            gender_map[j] = np.mean(gender_map[j])
    dwg = svgwrite.Drawing(size=(10, 10))
    dwg.add(dwg.rect(size=(10, 10), fill='white'))

    for k in range(100):
        line = k // 10
        col = k % 10
        if gender_map[k]:
            dwg.add(dwg.rect(insert=(col, line),
                             size=(1, 1),
                             fill=svgwrite.rgb(gender_map[k] * 100, 0, (1-gender_map[k]) * 100, mode='%')))

    dwg.saveas('gender.svg')


def sort_mps_d(mps, props, w, d):
    for i in range(349):
        ind = find_closest_w_row(props[i, :], w)
        d[mps[i]].append(ind)
    return d


def create_mps_dict(mps):
    mpsex = np.loadtxt('data/mpsex.dat', delimiter='\n', dtype='i', skiprows=2)
    mpparty = np.loadtxt('data/mpparty.dat', delimiter='\n', dtype='i', skiprows=3)
    mpdistrict = np.loadtxt('data/mpdistrict.dat', delimiter='\n', dtype='i')
    d = dict()
    for i in range(len(mps)):
        d[mps[i]] = [mpsex[i], mpparty[i], mpdistrict[i]]
    return d


def move_neurons(eta, eta_n, props, size, w):
    for i in range(349):
        ind = find_closest_w_row(props[i, :], w)
        w = update_w(props[i, :], w, ind, eta, eta_n, size)


def main():
    props = import_votes()
    w = train(props)
    mps = import_mps()
    d = create_mps_dict(mps)
    sorted_mps, pos = sort_mps(mps, props, w)
    gender_map(sorted_mps, pos, d)
    print(d)
    # for mp in d.keys():
    #     line = d[mp][3] // 10
    #     col = d[mp][3] % 10
    #     plt.plot(d[mp][3])


if __name__ == "__main__":
    main()
