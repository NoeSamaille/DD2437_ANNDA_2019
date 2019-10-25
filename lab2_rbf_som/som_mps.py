import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from matplotlib.lines import Line2D


def create_weight_matrix():
    w = np.random.uniform(0, 1, (100, 31))
    return w


def find_closest_w_row(x, w):
    d = np.zeros(100)
    for i in range(100):
        d[i] = euclidean(x, w[i, :])
    return np.argmin(d)


def is_neighbour(i, ind, size):
    """
    :param i: node index to check if it is a neighbour
    :param ind: checking if i is a neighbour of this node ind.
    :param size: size of neighbourhood. Ex size = 1 means i-1 and i+1 are neighbours.
    :return: True if it is a neighbour, False otherwise
    """
    y_i = i // 10
    x_i = i % 10
    y_ind = ind / 10
    x_ind = ind % 10
    return np.abs(y_ind - y_i) + np.abs(x_ind - x_i) <= size


def update_w(x, w, ind, eta, eta_n, size):
    """
    updates winner and its neighbours
    :param w:
    :param ind:
    :return:
    """
    for i in range(100):
        if i == ind:
            w[i, :] += eta * (x - w[i, :])
        else:
            w[i, :] += eta_n * is_neighbour(i, ind, size) * (x - w[i, :])
    return w


def move_neurons(eta, eta_n, votes, size, w):
    for i in range(349):
        ind = find_closest_w_row(votes[i, :], w)
        w = update_w(votes[i, :], w, ind, eta, eta_n, size)


def train(votes, eta=0.2, nb_epochs=20):
    alpha = 0.5
    eta_n = eta / 2
    w = create_weight_matrix()
    neighbourhood_size = 20
    for i in range(nb_epochs):
        print("Epoch", i)
        print("-> n:", neighbourhood_size)
        print("-> lr:", eta)
        print("-> lrn:", eta_n)
        move_neurons(eta, eta_n, votes, neighbourhood_size, w)
        # reduce learning rate and neighbourhood
        eta = alpha * eta**(i / nb_epochs)
        eta_n = alpha * eta_n**(i / nb_epochs)
        neighbourhood_size = round(20*((nb_epochs-i)/nb_epochs))
    return w


def analyse(votes, w):
    genders = np.loadtxt('data/mpsex.dat', comments="%")
    parties = np.loadtxt('data/mpparty.dat', comments="%")
    districts = np.loadtxt('data/mpdistrict.dat', comments="%")
    pos = np.zeros(349)
    genders_out = [[] for i in range(100)]
    parties_out = [[] for i in range(100)]
    districts_out = [[] for i in range(100)]

    for i in range(len(pos)):
        ind = find_closest_w_row(votes[i, :], w)
        pos[i] = ind
        genders_out[ind].append(genders[i])
        parties_out[ind].append(parties[i])
        districts_out[ind].append(districts[i])
    unique, counts = np.unique(pos, return_counts=True)

    # Plot Genders
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    for i in range(len(unique)):
        y = unique[i] // 10
        x = unique[i] % 10
        mean_gender = np.mean(genders_out[int(unique[i])])
        circle = plt.Circle((x+1, y+1), counts[i]/100, color=(mean_gender, 0, (1-mean_gender)))
        ax.add_artist(circle)
    plt.ylim(0, 11)
    plt.xlim(0, 11)
    plt.title("Genders (blue for male, red for female)")
    fig.savefig('images/mps_gender.png')

    # Plot Parties
    colors = ['orange', 'r', 'y', 'm', 'c', 'k', 'g', 'b']
    labels = ["no p", "m", "fp", "s", "v", "mp", "kd", "c"]
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    for i in range(len(unique)):
        y = unique[i] // 10
        x = unique[i] % 10
        parties, parties_count = np.unique(parties_out[int(unique[i])], return_counts=True)
        main_party = int(parties[np.argmax(parties_count)])
        circle = plt.Circle((x+1, y+1), counts[i]/100, color=colors[main_party])
        ax.annotate(str(int(np.max(parties_count)/len(parties_out[int(unique[i])])*100)) + "%", xy=(x+1, y+1), fontsize=7, ha="center")
        ax.add_artist(circle)
    legend_elements = []
    for i in range(8):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=labels[i],
                          markerfacecolor=colors[i], markersize=5))
    ax.legend(handles=legend_elements, loc='lower left')
    plt.ylim(0, 11)
    plt.xlim(-2, 11)
    plt.title("Dominating parties & their proportions")
    fig.savefig('images/mps_parties.png')

    # Plot Districts
    fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
    for i in range(len(unique)):
        y = unique[i] // 10
        x = unique[i] % 10
        districts = np.unique(districts_out[int(unique[i])])
        circle = plt.Circle((x+1, y+1), counts[i]/100, color='c')
        ax.annotate(str(int(len(districts))), xy=(x+1, y+1), fontsize=7, ha="center")
        ax.add_artist(circle)
    plt.ylim(0, 11)
    plt.xlim(0, 11)
    plt.title("Number of district per output")
    fig.savefig('images/mps_district.png')


def import_votes():
    return np.loadtxt('data/votes.dat', comments="%", delimiter=',').reshape((349, 31))


def main():
    votes = import_votes()
    w = train(votes)
    analyse(votes, w)


if __name__ == "__main__":
    main()