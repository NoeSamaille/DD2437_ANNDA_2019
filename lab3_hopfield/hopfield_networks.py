import matplotlib.pyplot as plt
import numpy as np
import itertools


def energy(W, x):
    return -np.sum(W*np.outer(x, x))


def energy_tracking():
    """
    looks at energy evolution with hebbian or random W
    calls en_recall for better performances
    :return:
    """
    pictures = import_pict()
    patterns, distorted_patterns = np.split(pictures, [9])
    # W = learn(patterns[:3])

    # np.random.seed(0)
    N = np.shape(patterns)[1]
    W = np.random.normal(0, 1, (N, N))
    for i in range(N):
        W[i, i] = 0
    W = 0.5 * (W + np.transpose(W))

    avg_conv = np.zeros(50)
    for i in range(50):
        rec, avg_conv[i] = en_recall(W, distorted_patterns[0])
    print(np.mean(avg_conv), np.std(avg_conv))
    #pbm('rand_w_p11', rec)


def en_recall(W, act, max_iter=100):
    """
    Actually the same function as recall and update neurons, but stripped from default arguments
    to improve performances
    """
    prev_act = act
    iter_en = np.zeros(max_iter)
    order = np.arange(len(act))

    for i in range(max_iter):
        np.random.shuffle(order)
        new_act = np.copy(prev_act)

        for j in order:
                    new_act[j] = np.sign(np.sum(W[j, :] * new_act))

        if np.array_equal(new_act, prev_act):
            # plt.plot(np.arange(0, i, 1), iter_en[:i])
            # plt.title("Energy depending on iterations")
            # plt.xlabel("Iterations")
            # plt.ylabel("Energy")
            # plt.show()

            return new_act, i
        prev_act = new_act
        iter_en[i] = energy(W, new_act)
    # plt.plot(np.arange(0, i, 1), iter_en[:i])
    # plt.title("Energy depending on iterations")
    # plt.xlabel("Iterations")
    # plt.ylabel("Energy")
    # plt.show()
    # print(iter_en)
    print("fail")
    return i


def recall(W, act, max_iter=10000, little_model=False, show_step=False, show_pic=False):
    """

    :param W: matrix
    :param act: pattern line vector, initial activation
    :param max_iter:
    :param little_model:
    :param show_step:
    :return:
    """
    prev_prev_act = None
    prev_act = act
    for i in range(max_iter):
        # print(i)
        new_act = update_neurons(W, prev_act, little_model=little_model, show_step=show_step, show_pic=show_pic)
        if np.array_equal(new_act, prev_act):
            return [new_act]
        if little_model and np.array_equal(new_act, prev_prev_act):
            return [new_act, prev_act]
        prev_prev_act = prev_act
        prev_act = new_act
    return new_act


def update_neurons(W, act, little_model=True, show_step=False, show_pic=False):
    if little_model:
        return np.sign(np.sum(W*act, axis=1))
    activations = np.copy(act)
    order = np.arange(len(act))
    np.random.shuffle(order)
    for i in order:
        if show_step and i%100 == 0:
            if show_pic:
                print_pic(activations)
            print(energy(W, activations))
        activations[i] = np.sign(np.sum(W[i,:]*activations))
    return activations


def sparse_recall(W, act, theta=1, max_iter=50):
    old = np.copy(act)
    new = sparse_update_neurons(W, old, theta=theta)
    tour = 1
    while not np.array_equal(old, new) or tour >= max_iter:
        new, old = sparse_update_neurons(W, new, theta=theta), new
        tour += 1
    return new


def sparse_update_neurons(W, act, theta=1):
    res = np.copy(act)
    for i in range(len(act)):
        res[i] = 0.5 + 0.5 * np.sign(np.dot(W[i], act) - theta)
    return res


def learn(patterns, W=None, nodiag=True, scale=True):
    P = np.shape(patterns)[0] # Number of patterns
    N = np.shape(patterns)[1] # Number of units
    if W is None:
        W = np.zeros((N, N))
    for u in range(P):
        W += np.outer(patterns[u], patterns[u])
    if nodiag:
        W -= np.diag(np.diag(W)) # Remove self-connections

    if scale:
        return W/N
    else:
        return W


def sparse_learn(patterns, activity):
    P = np.shape(patterns)[0] # Number of patterns
    N = np.shape(patterns)[1] # Number of units
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for u in range(P):
                W[i][j] += (patterns[u][i]-activity)*(patterns[u][j]-activity)
    W -= np.diag(np.diag(W)) # Remove self-connections
    return W


def attractors(W, pattern_dim):
    res = 0
    attr = []
    for pattern in list(itertools.product([-1, 1], repeat=pattern_dim)):
        if np.array_equal(pattern, update_neurons(W, pattern)):
            attr.append(pattern)
            res += 1
    return res, attr


def distorted_patterns():
    nb_iter = 1
    correct_rec = np.zeros((3, nb_iter))
    patterns = np.array([
        [-1, -1,  1, -1,  1, -1, -1,  1],
        [-1, -1, -1, -1, -1,  1, -1, -1],
        [-1,  1,  1, -1, -1,  1, -1,  1]])
    W = learn(patterns)
    distorted_patterns = np.array([
        [1, -1,  1, -1,  1, -1, -1,  1],
        [1,  1, -1, -1, -1,  1, -1, -1],
        [1,  1,  1, -1,  1,  1, -1,  1]])
    for i in range(len(distorted_patterns)):
        for j in range(nb_iter):
            rec = recall(W, distorted_patterns[i])
            correct_rec[i][j] = np.array_equal(patterns[i], rec[0])
            if nb_iter == 1:
                print(patterns[i], energy(W, patterns[i]))
                print("->", recall(W, distorted_patterns[i]), energy(W, distorted_patterns[i]))

    for k in range(3):
        print("xd" + str(k), np.mean(correct_rec[k]), np.std(correct_rec[k]))


def random_weights():
    W = np.random.normal(0, 1, size=(8, 8))
    # print(recall(W, [1, 1,  1, -1,  1, -1, 1,  1], little_model=False, show_step=True))
    print(attractors(W, 8))


def import_pict():
    return np.reshape(np.loadtxt('pict.dat', delimiter=','), (11, 1024))


def print_pic(pic):
    # Build image
    img = np.zeros((32, 32, 3))
    for i in range (32):
        for j in range(32):
            img[i][j] = [0, 0, 0] if pic[i*32+j]==-1 else [1, 1, 1]
    plt.imshow(img)
    plt.show()


def noise_stats(nb_patterns=3, noise_props=[0.05, 0.15, 0.25, 0.35, 0.45]):
    pictures = import_pict()
    patterns, distorted_patterns = np.split(pictures, [9])
    W = learn(patterns[:nb_patterns])
    n = len(patterns[2])
    nb_iter = 50
    scores = np.zeros((nb_patterns, len(noise_props)))
    for i in range(len(noise_props)):
        print("Noise prop :", noise_props[i])
        for p in range(nb_patterns):
            for _ in range(nb_iter):
                noisy_p = np.copy(patterns[p])
                indexes = np.random.choice(n, int(noise_props[i]*n), replace=False)
                for j in indexes:
                    noisy_p[j] *= -1
                if (np.array_equal(recall(W, noisy_p, little_model=True)[0], patterns[p])):
                    scores[p][i] += 1
    for p in range(nb_patterns):
        plt.plot(noise_props, scores[p]/nb_iter, label="p"+str(p+1))
    plt.ylim(0, 1)
    plt.xlabel("Noise proportion")
    plt.ylabel("Network accuracy (%)")
    plt.title("Network accuracy for different amount of noise ("+str(nb_patterns)+" memories)")
    plt.legend(loc="upper right")
    plt.show()


def random_pics_capacity_stats():
    nb_patterns = np.arange(5, int(0.138*1024)+5, 5)
    repeat = 10
    n = 1024
    noise_amount = int(0.05*n)
    stats = np.zeros((len(nb_patterns), 4))
    for i in range(len(nb_patterns)):
        scores = []
        stability_scores = []
        for _ in range(repeat):
            patterns = np.sign(np.random.uniform(-1, 1, (nb_patterns[i],n)))
            W = learn(patterns, nodiag=False)
            score = 0
            stability_score = 0
            for p in range(len(patterns)):
                noisy_p = np.copy(patterns[p])
                indexes = np.random.choice(n, noise_amount, replace=False)
                for j in indexes:
                    noisy_p[j] *= -1
                if (np.array_equal(recall(W, noisy_p, little_model=True)[0], patterns[p])):
                    score += 1
                if np.array_equal(update_neurons(W, patterns[p]), patterns[p]):
                    stability_score += 1
            scores.append(score/nb_patterns[i])
            stability_scores.append(stability_score/nb_patterns[i])
        stats[i][0] = np.average(scores)
        print(nb_patterns[i], ":", stats[i][0])
        stats[i][1] = np.std(scores)
        stats[i][2] = np.average(stability_scores)
        stats[i][3] = np.std(stability_scores)
    plt.errorbar(nb_patterns, stats[:, 0], yerr=stats[:, 1], label="Recall score (5% noise)")
    plt.errorbar(nb_patterns, stats[:, 2], yerr=stats[:, 3], label="Stability score")
    plt.ylim(0, 1)
    plt.xlabel("Number of learned patterns (memories)")
    plt.ylabel("Score")
    plt.title("Newtwork score depending on number of memories")
    plt.legend(loc="lower left")
    plt.show()


def add_noise(noise_prop, pattern):
    n = len(pattern)
    noisy_p = np.copy(pattern)
    indexes = np.random.choice(n, int(noise_prop * n), replace=False)
    for j in indexes:
        noisy_p[j] *= -1
    return noisy_p


def create_pbm():
    """
    given picture, writes pbm for the pic
    :return:
    """
    pictures = import_pict()
    for i in range(11):
        pbm("pic" + str(i + 1), pictures[i])


def pbm(name, pic):
    with open(name + ".pbm", "w") as f:
        f.write("P1\n32 32\n")
        picture = np.reshape(pic, (32, 32))
        for j in range(32):
            for k in range(32):
                if picture[j][k] == 1:
                    f.write("1 ")
                else:
                    f.write("0 ")
            f.write("\n")
    

def combinations():
    """
    sandbox function for testing the adding of several pictures
    :return:
    """
    pictures = import_pict()
    patterns, distorted_patterns = np.split(pictures, [9])
    W = learn(patterns[:3])
    recalled_pic = recall(W, patterns[3], little_model=False, show_step=True)
    recalled_pic = recalled_pic[0]
    recalled_pic[recalled_pic == -1] = 0
    p1 = pictures[0]
    # p4 = pictures[1]
    # p4[p4 == -1] = 0
    # # p1 = -p1
    p1[p1 == -1] = 0
    # p3 = pictures[2]
    # p3[p3 == -1] = 0
    p = np.logical_xor(p1, recalled_pic)
    # # p = np.logical_xor(p, p4)
    # print(p)
    pbm('test', p)


def capacity():
    """
    manually search for how many patterns can be stored
    :return:
    """
    storage = 4
    noise_rate = 0.4
    pictures = import_pict()
    patterns, distorted_patterns = np.split(pictures, [9])
    p = np.array([patterns[2], patterns[3], patterns[4], patterns[5]])
    # p = np.array([patterns[1], patterns[4], patterns[7], patterns[8]])
    W = learn(p)

    for i in range(storage):
        noisy_pattern = add_noise(noise_rate, p[i])
        pic = recall(W, noisy_pattern)[0]
        if np.array_equal(pic, p[i]) is False:
            print("pic", i, "not correctly recalled")
        else:
            print("pic", i, "correctly recalled")
        print_pic(pic)


def hundred_unit():
    """
    trains 300 patterns on a 100 unit network
    :return:
    """
    # patterns = np.random.choice([-1, 1], (300, 10), replace=True)
    W = np.zeros((10, 10))
    stable = np.zeros(300)
    f1_stable = np.zeros(300)
    patterns = np.sign(0.5 + np.random.normal(0, 1, size=(300, 10)))

    for i in range(300):
        W = learn(patterns[:i+1], nodiag=True, scale=True)
        for j in range(i+1):
            p = patterns[j]
            out = recall(W, p, max_iter=1)[0]

            # distortion
            flip1 = add_noise(0.1, patterns[j])
            flip1 = recall(W, flip1, max_iter=50)[0]

            if np.array_equal(patterns[j], flip1):
                f1_stable[i] += 1
                # print("pattern", j, "recalled for", i, "stored images")

            if np.array_equal(patterns[j], out):
                stable[i] += 1
                print(patterns[j])

    x = np.arange(1, 301, 1)
    plt.plot(x, stable, x, f1_stable)
    plt.xlabel("Number of patterns added to the weight matrix")
    plt.ylabel("Number of stable patterns")
    plt.title("Number of stable patterns depending on number of patterns added to the weight matrix")
    plt.legend(["No distortion", "1 flipped unit"])
    plt.show()


def sparse_patterns():
    activity = 0.01
    n = 100
    thetas = np.arange(0, 2, 0.2)
    noise_qty = int(0.05*n)
    repeat = 5
    stats = np.zeros((len(thetas),))
    for i in range(len(thetas)):
        print("theta", thetas[i])
        for nb_patterns in range(2, 14, 2):
            for _ in range(3):
                patterns = np.zeros((nb_patterns, n))
                indexes = np.random.choice(n*nb_patterns, int(activity*n*nb_patterns), replace=False)
                for index in indexes:
                    n_pattern = index // n
                    n_unit = index % n
                    patterns[n_pattern][n_unit] = 1
                W = sparse_learn(patterns, activity)
                score = 0
                for j in range(nb_patterns):
                    act = sparse_update_neurons(W, patterns[j], theta=thetas[i])
                    if np.array_equal(patterns[j], act):
                        score += 1
                if score == nb_patterns:
                    stats[i] = nb_patterns
    plt.plot(thetas, stats, label="Number of patterns")
    plt.xlabel("Theta")
    plt.ylabel("Number of patterns")
    plt.title("Number of patterns the network can store, depending on the bias (p="+str(activity)+")")
    plt.legend(loc="lower left")
    plt.show()


def main():
    # random_pics_capacity_stats()
    # hundred_unit()
    sparse_patterns()
    # capacity()
    # pictures = import_pict()
    # patterns, distorted_patterns = np.split(pictures, [9])
    # W = learn(patterns[:3])
    # n = len(patterns[2])
    # noisy_p = np.copy(patterns[2])
    # indexes = np.random.choice(n, int(0.8*n), replace=False)
    # for j in indexes:
    #     noisy_p[j] *= -1
    # print_pic(patterns[2])
    # print_pic(noisy_p)
    # print_pic(recall(W, noisy_p, little_model=True)[0])


if __name__ == "__main__":
    main()