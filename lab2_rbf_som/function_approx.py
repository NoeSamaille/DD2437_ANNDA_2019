import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from pcn import pcn_train, pcn_predict
from time import time


def predict(inputs, v, w, sigma, nb_rbf):
    inputs_t = np.transpose(inputs)
    nb_data = np.shape(inputs)[0] # Number of input patterns
    # Compute hidden nodes activations
    hidden_activations = np.zeros((nb_data, nb_rbf))
    for j in range(nb_rbf):
        for i in range(nb_data):
            # Apply RBF transfer function (gaussian)
            dist = np.sqrt(np.dot(np.transpose((inputs_t[:, i] - v[:, j])), (inputs_t[:, i] - v[:, j])))
            hidden_activations[i, j] = np.exp((-(dist)**2)/(2*sigma**2))
    # Return outputs
    return np.dot(hidden_activations, w)


def place_rbfs(inputs, nb_rbf, cl=False, share_win=False, l_rate=0.4, sigma=None):
    inputs_t = np.transpose(inputs)
    nb_data, input_dim = np.shape(inputs) # Number of input patterns, input patterns dimension
    v = np.zeros((input_dim, nb_rbf)) # RBF weights
    # Place RBFs to random datapoints
    indices = np.arange(nb_data)
    np.random.shuffle(indices)
    for i in range(nb_rbf):
        v[:, i] = inputs[indices[i], :]
    if cl is True:
        # Loop over the datapoints
        indices = np.arange(nb_data)
        np.random.shuffle(indices)
        nb_winners = 3
        for i in range(len(indices)):
            # Select the closest rbf
            #min_rbf = 0
            #min_dist = float('+inf')
            if share_win is True:
                d = (inputs_t[:, i] - v) * (inputs_t[:, i] - v)  # computes distances from a given points to all rbfs
                winners_ind = np.argpartition(d[0, :], -nb_winners)[:nb_winners]  # retrieves indexes of the nb_winners, nb_winners acts as a pivotal index
                winners_ind = winners_ind[np.argsort(d[0, winners_ind])]  # winners_ind is reorder by shorter distances
                #winners = d[0, winners_ind]
            else:
                # keep this section for the moment
                min_dist = 10000000
                for j in range(nb_rbf):
                    cur_dist = np.dot(np.transpose((inputs_t[:, i] - v[:, j])), (inputs_t[:, i] - v[:, j]))
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        min_rbf = j
            # Update weight
            if share_win is True:
                for k in winners_ind:
                    v[:, k] += l_rate * (np.transpose(inputs)[:, i] - v[:, k])
            else:
                v[:, min_rbf] += l_rate * (np.transpose(inputs)[:, i] - v[:, min_rbf])
    if sigma is None:
        # Compute sigma (size of the gaussians) so that rbf radius overlap
        max_d = 0
        for i in range(nb_rbf):
            for j in range(i+1, nb_rbf):
                cur_d = np.absolute(v[:, i]-v[:, j])
                if cur_d > max_d:
                    max_d = cur_d
        sigma = max_d/np.sqrt(2*nb_rbf)
    return v, sigma


def batch_train(inputs, targets, nb_rbf, sigma=None, rand_rbfs=False, cl=False, l_rate=0.1, share_win=False):
    """
    Inputs are unidimensional points
    :param inputs:
    :param targets:
    :param nb_rbf:
    :return:
    """
    inputs_t = np.transpose(inputs)
    nb_data, input_dim = np.shape(inputs) # Number of input patterns, input patterns dimension
    v, sigma = place_rbfs(inputs, nb_rbf, cl, share_win=share_win, l_rate=l_rate, sigma=sigma)
    # v, sigma = np.reshape(np.arange(0, 2*math.pi+math.pi/4, math.pi/4), (input_dim, nb_rbf)), 2
    # Compute hidden nodes activations
    hidden_activations = np.zeros((nb_data, nb_rbf))
    for j in range(nb_rbf):
        for i in range(nb_data):
            # Apply RBF transfer function (gaussian)
            dist = np.sqrt(np.dot(np.transpose((inputs_t[:, i] - v[:, j])), (inputs_t[:, i] - v[:, j])))
            hidden_activations[i, j] = np.exp((-(dist)**2)/(2*sigma**2))
    # Train the output weights
    w = np.dot(np.linalg.pinv(hidden_activations), targets)
    return v, w, sigma


def seq_train(inputs, targets, nb_rbf, sigma=None, l_rate=0.1, nb_epochs=400, cl=False):
    nb_data, input_dim = np.shape(inputs) # Number of input patterns, input patterns dimension
    v, sigma = place_rbfs(inputs, nb_rbf, cl, l_rate)
    # Init w
    w = np.random.normal(0, 0.1, (nb_rbf, np.shape(targets)[1]))
    inputs = np.transpose(inputs)
    targets = np.transpose(targets)
    hidden_activations = np.zeros((nb_rbf, input_dim))
    for epoch in range(nb_epochs):
        inputs = np.concatenate((inputs, targets))
        inputs = np.transpose(inputs)
        np.random.shuffle(inputs)
        inputs = np.transpose(inputs)
        inputs, targets = np.split(inputs, [1])
        #print("Epoch: ", epoch)
        # Loop over the input patterns
        for i in range(np.shape(inputs)[1]):
            # Compute hidden nodes activations for input pattern i
            for j in range(nb_rbf):
                # Apply RBF transfer function (gaussian)
                hidden_activations[j, :] = np.transpose(np.exp((-(inputs[:, i] - v[:,j])**2)/(2*sigma**2)))
            # Update the weights
            w = w + l_rate * (targets[:, i] - np.dot(np.transpose(hidden_activations), w)) * hidden_activations
    return v, w, sigma


def gen_data(noise=False, do_plot=False):
    # Generate training data
    train_x = np.arange(0, 2*math.pi+0.1, 0.1)
    f1_train_targets = np.sin(2*train_x)
    f2_train_targets = np.sign(f1_train_targets)
    # Generate testing data
    test_x = np.arange(0.05, 2*math.pi+0.05, 0.1)
    f1_test_targets = np.sin(2*test_x)
    f2_test_targets = np.sign(f1_test_targets)
    if noise:
        # Add noise to data
        train_x += np.random.normal(0, np.sqrt(0.1))
        test_x += np.random.normal(0, np.sqrt(0.1))
    # Plot training data
    if do_plot:
        plt.plot(train_x, f1_train_targets, label="sin(2x)")
        plt.plot(train_x, f2_train_targets, label="square(2x)")
        plt.legend(loc="lower left")
        plt.show()
    return np.reshape(train_x, (len(train_x), 1)), np.reshape(f1_train_targets, (len(f1_train_targets), 1)), np.reshape(f2_train_targets, (len(f2_train_targets), 1)), np.reshape(test_x, (len(test_x), 1)), np.reshape(f1_test_targets, (len(f1_test_targets), 1)), np.reshape(f2_test_targets, (len(f2_test_targets), 1))


def residual_err(predictions, targets):
    return np.sum(np.average(np.absolute(predictions - targets)))


def plot(x, targets, predictions):
    plt.plot(x, targets, label="targets")
    plt.plot(x, predictions, label="predictions")
    plt.legend(loc="lower left")
    plt.show()


def stats_nb_RBFs(start=2, stop=30, step=1, nb_iter=20, sin=True, square=True, noise=False, batch=True):
    train_x, sin_train_targets, square_train_targets, test_x, sin_test_targets, square_test_targets = gen_data(noise=noise)
    train_x_noise, _, _, test_x_noise, _, _ = gen_data(noise=True)
    nb_rbf = np.arange(start, stop, step)
    sin_stats = np.zeros((2, len(nb_rbf)))
    sin_stats_noise = np.zeros((2, len(nb_rbf)))
    square_stats = np.zeros((2, len(nb_rbf)))
    square_stats_noise = np.zeros((2, len(nb_rbf)))
    for i in range(len(nb_rbf)):
        #print("RBFs: ", nb_rbf[i])
        sin_err = np.zeros(nb_iter)
        square_err = np.zeros(nb_iter)
        sin_err_noise = np.zeros(nb_iter)
        square_err_noise = np.zeros(nb_iter)
        for j in range(nb_iter):
            #print("-> Iter: ", j)
            if sin:
                if batch:
                    v, w, sigma = batch_train(train_x, sin_train_targets, nb_rbf[i])
                else:
                    v, w, sigma = seq_train(train_x, sin_train_targets, nb_rbf[i])
                predictions = predict(test_x, v, w, sigma, nb_rbf[i])
                sin_err[j] = residual_err(predictions, sin_test_targets)

                v, w, sigma = seq_train(train_x_noise, sin_train_targets, nb_rbf[i])
                predictions = predict(test_x_noise, v, w, sigma, nb_rbf[i])
                sin_err_noise[j] = residual_err(predictions, sin_test_targets)
            if square:
                if batch:
                    v, w, sigma = batch_train(train_x, square_train_targets, nb_rbf[i])
                else:
                    v, w, sigma = seq_train(train_x, square_train_targets, nb_rbf[i])
                predictions = predict(test_x, v, w, sigma, nb_rbf[i])
                square_err[j] = residual_err(predictions, square_test_targets)

                v, w, sigma = seq_train(train_x_noise, square_train_targets, nb_rbf[i])
                predictions = predict(test_x_noise, v, w, sigma, nb_rbf[i])
                square_err_noise[j] = residual_err(predictions, square_test_targets)
        sin_stats[0][i] = np.average(sin_err)
        sin_stats[1][i] = np.std(sin_err)
        sin_stats_noise[0][i] = np.average(sin_err_noise)
        sin_stats_noise[1][i] = np.std(sin_err_noise)
        square_stats[0][i] = np.average(square_err)
        square_stats[1][i] = np.std(square_err)
        square_stats_noise[0][i] = np.average(square_err_noise)
        square_stats_noise[1][i] = np.std(square_err_noise)
    plt.title("Residual error depending on the number of RBFs")
    if sin:
        print("Min sin(2x) err: ", np.min(sin_stats[0]))
        plt.errorbar(nb_rbf, sin_stats[0], yerr=1.96 * sin_stats[1], label="sin(2x)")
        plt.errorbar(nb_rbf, sin_stats_noise[0], yerr=1.96 * sin_stats_noise[1], label="noise sin(2x)")
    if square:
        print("Min square(2x) err: ", np.min(square_stats[0]))
        plt.errorbar(nb_rbf, square_stats[0], yerr=1.96 * square_stats[1], label="square(2x)")
        plt.errorbar(nb_rbf, square_stats_noise[0], yerr=1.96 * square_stats_noise[1], label="noise square(2x)")
    plt.ylabel("Residual error")
    plt.xlabel("Number of RBF units")
    plt.legend(loc="upper right")
    plt.show()


def stats_width_RBFs(start=0.2, stop=3, step=0.2, nb_iter=20, sin=True, square=True):
    train_x, sin_train_targets, square_train_targets, test_x, sin_test_targets, square_test_targets = gen_data(noise=True)
    width_rbf = np.arange(start, stop, step)
    sin_stats_noise = np.zeros((2, len(width_rbf)))
    square_stats_noise = np.zeros((2, len(width_rbf)))
    for i in range(len(width_rbf)):
        print("RBFs: ", width_rbf[i])
        sin_err_noise = np.zeros(nb_iter)
        square_err_noise = np.zeros(nb_iter)
        for j in range(nb_iter):
            print("-> Iter: ", j)
            if sin:
                v, w, sigma = seq_train(train_x, sin_train_targets, 13, sigma=width_rbf[i])
                predictions = predict(test_x, v, w, sigma, 13)
                sin_err_noise[j] = residual_err(predictions, sin_test_targets)
            if square:
                v, w, sigma = seq_train(train_x, square_train_targets, 7, sigma=width_rbf[i])
                predictions = predict(test_x, v, w, sigma, 7)
                square_err_noise[j] = residual_err(predictions, square_test_targets)
        sin_stats_noise[0][i] = np.average(sin_err_noise)
        sin_stats_noise[1][i] = np.std(sin_err_noise)

        square_stats_noise[0][i] = np.average(square_err_noise)
        square_stats_noise[1][i] = np.std(square_err_noise)
    plt.title("Residual error depending on the width of RBFs")
    if sin:
        print("Min sin(2x) err: ", np.min(sin_stats_noise[0]))
        plt.errorbar(width_rbf, sin_stats_noise[0], yerr=1.96 * sin_stats_noise[1], label="noise sin(2x)")
    if square:
        print("Min square(2x) err: ", np.min(square_stats_noise[0]))
        plt.errorbar(width_rbf, square_stats_noise[0], yerr=1.96 * square_stats_noise[1], label="noise square(2x)")
    plt.ylabel("Residual error")
    plt.xlabel("RBF width size")
    plt.legend(loc="upper right")
    plt.show()


def stats_eta_RBFs(nb_iter=20, sin=True, square=True):
    train_x, sin_train_targets, square_train_targets, test_x, sin_test_targets, square_test_targets = gen_data(noise=True)
    eta = np.array([0.001, 0.005, 0.01, 0.015, 0.03, 0.05, 0.08, 0.1])
    sin_stats_noise = np.zeros((2, len(eta)))
    square_stats_noise = np.zeros((2, len(eta)))
    for i in range(len(eta)):
        print("Eta: ", eta[i])
        sin_err_noise = np.zeros(nb_iter)
        square_err_noise = np.zeros(nb_iter)
        for j in range(nb_iter):
            print("-> Iter: ", j)
            if sin:
                v, w, sigma = seq_train(train_x, sin_train_targets, 13, l_rate=eta[i])
                predictions = predict(test_x, v, w, sigma, 13)
                sin_err_noise[j] = residual_err(predictions, sin_test_targets)
            if square:
                v, w, sigma = seq_train(train_x, square_train_targets, 7, l_rate=eta[i])
                predictions = predict(test_x, v, w, sigma, 7)
                square_err_noise[j] = residual_err(predictions, square_test_targets)
        sin_stats_noise[0][i] = np.average(sin_err_noise)
        sin_stats_noise[1][i] = np.std(sin_err_noise)

        square_stats_noise[0][i] = np.average(square_err_noise)
        square_stats_noise[1][i] = np.std(square_err_noise)
    plt.title("Residual error depending on eta")
    if sin:
        print("Min sin(2x) err: ", np.min(sin_stats_noise[0]))
        plt.errorbar(eta, sin_stats_noise[0], yerr=1.96 * sin_stats_noise[1], label="noise sin(2x)")
    if square:
        print("Min square(2x) err: ", np.min(square_stats_noise[0]))
        plt.errorbar(eta, square_stats_noise[0], yerr=1.96 * square_stats_noise[1], label="noise square(2x)")
    plt.ylabel("Residual error")
    plt.xlabel("Learning rate eta")
    plt.legend(loc="upper right")
    plt.show()


def test_on_clean():
    nb_iter = 20
    train_x_noise, sin_train_targets, square_train_targets, test_x_noise, sin_test_targets, square_test_targets = gen_data(noise=True)

    train_x, _, _, test_x, _, _ = gen_data(noise=False)
    sin_clean_err = np.zeros(nb_iter)
    sin_noise_err = np.zeros(nb_iter)
    square_clean_err = np.zeros(nb_iter)
    square_noise_err = np.zeros(nb_iter)
    
    for i in range(nb_iter):
        v_square_noise, w_square_noise, sigma_square_noise = batch_train(train_x_noise, square_train_targets, 7)
        v_sin_noise, w_sin_noise, sigma_sin_noise = batch_train(train_x_noise, sin_train_targets, 13)
        
        v_square, w_square, sigma_square = batch_train(train_x, square_train_targets, 7)
        v_sin, w_sin, sigma_sin = batch_train(train_x, sin_train_targets, 13)
    
        predictions_square = predict(test_x, v_square, w_square, sigma_square, 7)
        predictions_sin = predict(test_x, v_sin, w_sin, sigma_sin, 13)
        predictions_square_noise = predict(test_x, v_square_noise, w_square_noise, sigma_square_noise, 7)
        predictions_sin_noise = predict(test_x, v_sin_noise, w_sin_noise, sigma_sin_noise, 13)
        sin_clean_err[i] = residual_err(predictions_sin, sin_test_targets)
        sin_noise_err[i] = residual_err(predictions_sin_noise, sin_test_targets)
        square_clean_err[i] = residual_err(predictions_square, square_test_targets)
        square_noise_err[i] = residual_err(predictions_square_noise, square_test_targets)
    
       # plt.title('Batch learning approximating sin(2x) with 13 RBFs')
       # plt.xlabel('x')
       # plt.ylabel('y')
       # plt.plot(test_x, sin_test_targets, label='target')
       # plt.plot(test_x, predictions_sin, label='trained on clean set')
       # plt.plot(test_x, predictions_sin_noise, label='trained on noised set')
       # plt.legend(loc="lower left")
       # plt.show()
       #  print('trained on clean set error', residual_err(predictions_sin, sin_test_targets))
       #  print('trained on noised set error', residual_err(predictions_sin_noise, sin_test_targets))
       #
       # plt.clf()
       # plt.title('Batch learning approximating square(2x) with 13 RBFs')
       # plt.xlabel('x')
       # plt.ylabel('y')
       # plt.plot(test_x, square_test_targets, label='target')
       # plt.plot(test_x, np.sign(predictions_square), label='trained on clean set')
       # plt.plot(test_x, np.sign(predictions_square_noise), label='trained on noised set')
       # plt.legend(loc="lower left")
       # plt.show()
       # print('trained on clean set error', residual_err(predictions_square, square_test_targets))
       # print('trained on noised set error', residual_err(predictions_square_noise, square_test_targets))
    print("trained on clean set error sin", np.mean(sin_clean_err), "sd", np.std(sin_clean_err))
    print("trained on noised set error sin", np.mean(sin_noise_err), "sd", np.std(sin_noise_err))    
    print("trained on clean set error square", np.mean(square_clean_err), "sd", np.std(square_clean_err))
    print("trained on noised set error square", np.mean(square_noise_err), "sd", np.std(square_noise_err))


def pcn_tune():
    """
    empirically finds right lr and epochs
    :return:
    """
    #square: 800 epochs, learning rate of 0.04
    epochs = 800
    lr = 0.04
    train_x_noise, sin_train_targets, square_train_targets, test_x_noise, sin_test_targets, square_test_targets = gen_data(
        noise=True)
    
    train_x_noise = np.transpose(train_x_noise)
    train_x_noise = np.concatenate((train_x_noise, np.ones((1, len(train_x_noise[0])))))
    square_train_targets = np.transpose(square_train_targets)
    sin_train_targets = np.transpose(sin_train_targets)
    
    v_square_pcn, w_square_pcn, error_square = pcn_train(train_x_noise, 
                                                         square_train_targets, 
                                                         epochs, lr, 7, 1, 1)
    v_sin_pcn, v_sin_pcn, error_sin = pcn_train(train_x_noise, 
                                                sin_train_targets, 
                                                epochs, lr, 13, 1, 1)
    x = np.arange(0, epochs, 1)
    plt.plot(x, error_square, label='square')
    plt.plot(x, error_sin, label='sin')
    plt.legend(loc="upper left")


def compare():
    
    avg_pcn = np.zeros(20)
    avg_rbf = np.zeros(20)
    
    for i in range(20):
        train_x_noise, sin_train_targets, square_train_targets, test_x_noise, sin_test_targets, square_test_targets = gen_data(
        noise=True)
        start_rbf = time()
        v_square_noise, w_square_noise, sigma_square_noise = batch_train(train_x_noise, square_train_targets, 7)
        end_rbf = time()
        avg_rbf[i] = end_rbf - start_rbf
        # print("rbf training lasted", end_rbf - start_rbf)

        v_sin_noise, w_sin_noise, sigma_sin_noise = batch_train(train_x_noise, sin_train_targets, 13)
        
        train_x_noise_pcn = np.transpose(train_x_noise)
        train_x_noise_pcn = np.concatenate((train_x_noise_pcn, np.ones((1, len(train_x_noise_pcn[0])))))
        
        start_pcn = time()
        v_square_pcn, w_square_pcn, error_square = pcn_train(train_x_noise_pcn,
                                                             np.transpose(square_train_targets), 
                                                             800, 0.04, 7, 1, 1)
        end_pcn = time()
        avg_pcn[i] = end_pcn - start_pcn
    print("rbf lasted", np.mean(avg_rbf), "sd", np.std(avg_rbf))
    print("pcn lasted", np.mean(avg_pcn), "sd", np.std(avg_pcn))
        # print("pcn_training lasted", end_pcn - start_pcn)
    # v_sin_pcn, w_sin_pcn, error_sin = pcn_train(train_x_noise_pcn, np.transpose(sin_train_targets), 800, 0.04, 13, 1, 1)
    #
    # test_x_noise_pcn = np.transpose(test_x_noise)
    # test_x_noise_pcn = np.concatenate((test_x_noise_pcn, np.ones((1, len(test_x_noise_pcn[0])))))
    #
    # predictions_square_pcn, _ = pcn_predict(test_x_noise_pcn, v_square_pcn, w_square_pcn)
    #
    # predictions_sin_pcn, _ = pcn_predict(test_x_noise_pcn, v_sin_pcn, w_sin_pcn)
    #
    # predictions_square_noise = predict(test_x_noise, v_square_noise, w_square_noise, sigma_square_noise, 7)
    # predictions_sin_noise = predict(test_x_noise, v_sin_noise, w_sin_noise, sigma_sin_noise, 13)
    #
    # plt.title('Batch learning approximating sin(2x)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.plot(test_x_noise, sin_test_targets, label='target')
    # plt.plot(test_x_noise, np.transpose(predictions_sin_pcn), label='trained using batch pcn')
    # plt.plot(test_x_noise, predictions_sin_noise, label='trained using batch RBF')
    # plt.legend(loc='lower left')
    # plt.show()
    #
    # plt.clf()
    # plt.title('Batch learning approximating square(2x)')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.plot(test_x_noise, square_test_targets, label='target')
    # plt.plot(test_x_noise, np.transpose(predictions_square_pcn), label='trained using batch pcn')
    # plt.plot(test_x_noise, predictions_square_noise, label='trained using batch RBF')
    # plt.legend(loc='lower left')
    # plt.show()


def err_cl():
    lr = 0.1
    ep = 500
    train_x, sin_train_targets, square_train_targets, test_x, sin_test_targets, square_test_targets = gen_data(noise=True)
    err = []
    for i in range(100):
        v, w, sigma = batch_train(train_x, sin_train_targets, 16, cl=True, l_rate=lr, share_win=False)
        predictions = predict(test_x, v, w, sigma, 16)
        err.append(residual_err(predictions, sin_test_targets))
        #plot(test_x, sin_test_targets, predictions)
    print("Avg residual err with CL: ", np.average(err))
    print("-> std: ", np.std(err))
    for i in range(100):
        v, w, sigma = batch_train(train_x, sin_train_targets, 16, cl=True, l_rate=lr, share_win=True)
        predictions = predict(test_x, v, w, sigma, 16)
        err.append(residual_err(predictions, sin_test_targets))
        #plot(test_x, sin_test_targets, predictions)
    print("Avg residual err with CL, 3 winners", np.average(err))
    print("-> std: ", np.std(err))
    for i in range(100):
        v, w, sigma = batch_train(train_x, sin_train_targets, 16, cl=False, l_rate=lr, share_win=False)
        predictions = predict(test_x, v, w, sigma, 16)
        err.append(residual_err(predictions, sin_test_targets))
        #plot(test_x, sin_test_targets, predictions)
    print("Avg residual err without CL: ", np.average(err))
    print("-> std: ", np.std(err))


def two_dim_regression():
    train_data = np.genfromtxt("data/ballist.dat", delimiter=" ")
    train_inputs, train_targets = np.split(train_data, 2, axis=1)
    test_data = np.genfromtxt("data/balltest.dat", delimiter=" ")
    test_inputs, test_targets = np.split(test_data, 2, axis=1)

    lr = 0.1
    v, w, sigma = batch_train(train_inputs, train_targets, 16, cl=True, l_rate=lr, sigma=1)
    predictions = predict(test_inputs, v, w, sigma, 16)
    # print(residual_err(predictions, test_targets))
    ax = plt.axes(projection='3d')
    ax.scatter(test_inputs[:, 0], test_inputs[:, 1], test_targets[:, 0], label="Targets")
    ax.scatter(test_inputs[:, 0], test_inputs[:, 1], predictions[:, 0], label="Predictions")
    ax.set_xlabel('Angle')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Distance')
    plt.legend(loc="upper left")
    plt.show()
    ax = plt.axes(projection='3d')
    ax.scatter(test_inputs[:, 0], test_inputs[:, 1], test_targets[:, 1], label="Targets")
    ax.scatter(test_inputs[:, 0], test_inputs[:, 1], predictions[:, 1], label="Predictions")
    ax.set_xlabel('Angle')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Height')
    plt.legend(loc="upper left")
    plt.show()


def main():
    # stats_nb_RBFs(noise=True)
    two_dim_regression()
    # err_cl()
    

if __name__ == "__main__":
    main()
