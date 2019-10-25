import math
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.losses import mean_squared_error


# Saves the 1500 first values of the time series for computanional purposes
xs = np.zeros((1500))


def x(t):
    if t == 0.0:
        # Starting value
        xs[t] = 1.5
        return xs[t] 
    if t < 0.0:
        return 0
    # Compute x(t-1)
    xtm1 = xs[t-1] if t < 1500 and xs[t-1] != 0.0 else x(t-1)
    # Compute x(t-26)
    xtm26 = 0 if t < 26 else xs[t-26] if xs[t-26] != 0.0 else x(t-26)
    # Compute the result
    res = xtm1 + (0.2*xtm26/(1+xtm26**10)) - 0.1*xtm1
    if t < 1500:
        xs[t] = res
    return res


def plot_mg():
    t = np.arange(301, 1501, 1)
    X = np.zeros(len(t))
    for idx, time in enumerate(t):
        X[idx] = x(time)
    plt.plot(t, X)
    plt.show()


def split_inputs(input_patterns, targets, proportion):
    """
    Chose a ramdom sample of size (proportion*np.shape(input_patterns)[1]) of the 
    input patterns for testing

    Params:
        input_patterns: input patterns
        targets: target values matching the input patterns
        proportion: proportion of the testing set
    
    Returns:
        data_patterns: data set (training and validation)
        data_targets: data targets (training and validation)
        test_patterns: test set
        test_targets: test targets
    """
    _, m = np.shape(input_patterns)
    # Random sample amoung m of proportion (proportion * m)
    n_test = int(proportion * m)
    sample = np.random.choice(m, n_test, replace=False)
    # Mask matrix matching sample
    mask = np.zeros(m, dtype=bool)
    mask[sample] = True
    # Get test set and targets matching mask
    test_patterns = input_patterns[:, mask]
    test_targets = targets[mask]
    # Get data set and targets matching mask
    data_patterns = input_patterns[:, ~mask]
    data_targets = targets[~mask]

    return data_patterns, data_targets, test_patterns, test_targets


def compare_pcns():
    nb_iter = 50
    # Compare 2 layers architecture with 3 layers architecture
    losses_2l = np.zeros(nb_iter)
    losses_3l = np.zeros(nb_iter)
    for j in range(nb_iter):
        print('-> Iter: ', j, '/', nb_iter)
        # Inputs generation
        nb_data = 1200
        t = np.arange(301, 1501, 1)
        input_patterns = np.random.normal(0.0, 0.09, (6, nb_data))
        for i in range(nb_data):
            input_patterns[0][i] += x(t[i]-20)
            input_patterns[1][i] += x(t[i]-15)
            input_patterns[2][i] += x(t[i]-10)
            input_patterns[3][i] += x(t[i]-5)
            input_patterns[4][i] += x(t[i])
            input_patterns[5][i] = x(t[i]+5) # Targets
        input_patterns, targets = np.split(input_patterns, [5])
        # Split the inputs
        data_patterns, data_targets, test_patterns, test_targets = split_inputs(input_patterns, targets.flatten(), 0.17)
        # Build and train the 2 layers network
        nb_hidden_layers = 1
        nb_hidden_nodes = [5]
        model, _ = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes)
        losses_2l[j] = model.evaluate(np.transpose(test_patterns), np.transpose(test_targets), verbose=0)[0]
        nb_hidden_layers = 2
        nb_hidden_nodes = [5, 6]
        model, _ = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes)
        losses_3l[j] = model.evaluate(np.transpose(test_patterns), np.transpose(test_targets), verbose=0)[0]
    print('2 Layers networks')
    print('-> err: ', np.average(losses_2l))
    print('-> std: ', np.std(losses_2l))
    print('3 Layers networks')
    print('-> err: ', np.average(losses_3l))
    print('-> std: ', np.std(losses_3l))


def stats_reg():
    # Regularisation trengths
    reg_strengths = [0.00001, 0.0001, 0.001]
    # Noise standard deviations
    noise_sd = [0.03, 0.09, 0.18]
    # Stats
    avg_train_loss = np.zeros((len(noise_sd), len(reg_strengths)))
    std_train_loss = np.zeros((len(noise_sd), len(reg_strengths)))
    avg_eval_loss = np.zeros((len(noise_sd), len(reg_strengths)))
    std_eval_loss = np.zeros((len(noise_sd), len(reg_strengths)))
    avg_test_loss = np.zeros((len(noise_sd), len(reg_strengths)))
    std_test_loss = np.zeros((len(noise_sd), len(reg_strengths)))
    for ni in range(1, 2, 1):
        print('-> noise_sd: ', noise_sd[ni])
        # Inputs generation
        nb_data = 1200
        t = np.arange(301, 1501, 1)
        input_patterns = np.random.normal(0.0, noise_sd[ni], (6, nb_data))
        for i in range(nb_data):
            input_patterns[0][i] += x(t[i]-20)
            input_patterns[1][i] += x(t[i]-15)
            input_patterns[2][i] += x(t[i]-10)
            input_patterns[3][i] += x(t[i]-5)
            input_patterns[4][i] += x(t[i])
            input_patterns[5][i] = x(t[i]+5) # Targets
        input_patterns, targets = np.split(input_patterns, [5])
        # Split the inputs
        data_patterns, data_targets, test_patterns, test_targets = split_inputs(input_patterns, targets.flatten(), 0.17)
        # Build and train the network
        for nrs in range(len(reg_strengths)):
            print('--> reg strength: ', reg_strengths[nrs])
            train_loss = np.zeros(10)
            eval_loss = np.zeros(10)
            test_loss = np.zeros(10)
            for i in range(10):
                print('---> iter ', i)
                nb_hidden_layers = 2
                nb_hidden_nodes = [5, 6]
                model, history = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes)
                train_loss[i] = history.history['loss'][len(history.history['loss'])-1]
                eval_loss[i] = history.history['val_loss'][len(history.history['val_loss'])-1]
                test_loss[i] = model.evaluate(np.transpose(test_patterns), np.transpose(test_targets), verbose=0)[0]
            avg_train_loss[ni][nrs] = np.average(train_loss)
            print("avg train loss: ", avg_train_loss[ni][nrs])
            std_train_loss[ni][nrs] = np.std(train_loss)
            print("std train loss: ", std_train_loss[ni][nrs])
            avg_eval_loss[ni][nrs] = np.average(eval_loss)
            print("avg eval loss: ", avg_eval_loss[ni][nrs])
            std_eval_loss[ni][nrs] = np.std(eval_loss)
            print("std eval loss: ", std_eval_loss[ni][nrs])
            avg_test_loss[ni][nrs] = np.average(test_loss)
            print("avg test loss: ", avg_test_loss[ni][nrs])
            std_test_loss[ni][nrs] = np.std(test_loss)
            print("std test loss: ", std_test_loss[ni][nrs])
    print('Train losses', avg_train_loss)
    print('-> std', std_train_loss)
    print('Eval losses', avg_eval_loss)
    print('-> std', std_eval_loss)
    print('Test losses', avg_test_loss)
    print('-> std', std_test_loss)
    for ni in range(1,2,1):
        plt.errorbar(reg_strengths, avg_train_loss[ni], yerr=std_train_loss[ni], label='Train loss')
        plt.errorbar(reg_strengths, avg_eval_loss[ni], yerr=std_eval_loss[ni], label='Validation loss')
        plt.errorbar(reg_strengths, avg_test_loss[ni], yerr=std_test_loss[ni], label='Test loss')
        plt.title('NN loss depending on regularisation, noise std: ' + str(noise_sd[ni]))
        plt.ylabel('Loss')
        plt.xlabel('Regularisation factor')
        plt.legend(loc='upper right')
        plt.show()


def stats_hnodes_sec_hlayers():
    # Noise standard deviations
    noise_sd = [0.03, 0.09, 0.18]
    # Stats
    avg_train_loss = np.zeros((len(noise_sd), 8))
    std_train_loss = np.zeros((len(noise_sd), 8))
    avg_eval_loss = np.zeros((len(noise_sd), 8))
    std_eval_loss = np.zeros((len(noise_sd), 8))
    avg_test_loss = np.zeros((len(noise_sd), 8))
    std_test_loss = np.zeros((len(noise_sd), 8))
    for ni in range(len(noise_sd)):
        print('-> noise_sd: ', noise_sd[ni])
        # Inputs generation
        nb_data = 1200
        t = np.arange(301, 1501, 1)
        input_patterns = np.random.normal(0.0, noise_sd[ni], (6, nb_data))
        for i in range(nb_data):
            input_patterns[0][i] += x(t[i]-20)
            input_patterns[1][i] += x(t[i]-15)
            input_patterns[2][i] += x(t[i]-10)
            input_patterns[3][i] += x(t[i]-5)
            input_patterns[4][i] += x(t[i])
            input_patterns[5][i] = x(t[i]+5) # Targets
        input_patterns, targets = np.split(input_patterns, [5])
        # Split the inputs
        data_patterns, data_targets, test_patterns, test_targets = split_inputs(input_patterns, targets.flatten(), 0.17)
        # Build and train the network
        for hnsl in range(8):
            print('--> ', hnsl, ' hidden nodes')
            train_loss = np.zeros(10)
            eval_loss = np.zeros(10)
            test_loss = np.zeros(10)
            for i in range(10):
                print('---> iter ', i)
                nb_hidden_layers = 2
                nb_hidden_nodes = [5, hnsl+1]
                model, history = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes)
                train_loss[i] = history.history['loss'][len(history.history['loss'])-1]
                eval_loss[i] = history.history['val_loss'][len(history.history['val_loss'])-1]
                test_loss[i] = model.evaluate(np.transpose(test_patterns), np.transpose(test_targets), verbose=0)[0]
            avg_train_loss[ni][hnsl] = np.average(train_loss)
            print("avg train loss: ", avg_train_loss[ni][hnsl])
            std_train_loss[ni][hnsl] = np.std(train_loss)
            print("std train loss: ", std_train_loss[ni][hnsl])
            avg_eval_loss[ni][hnsl] = np.average(eval_loss)
            print("avg eval loss: ", avg_eval_loss[ni][hnsl])
            std_eval_loss[ni][hnsl] = np.std(eval_loss)
            print("std eval loss: ", std_eval_loss[ni][hnsl])
            avg_test_loss[ni][hnsl] = np.average(test_loss)
            print("avg test loss: ", avg_test_loss[ni][hnsl])
            std_test_loss[ni][hnsl] = np.std(test_loss)
            print("std test loss: ", std_test_loss[ni][hnsl])
    print('Train losses', avg_train_loss)
    print('-> std', std_train_loss)
    print('Eval losses', avg_eval_loss)
    print('-> std', std_eval_loss)
    print('Test losses', avg_test_loss)
    print('-> std', std_test_loss)
    for ni in range(len(noise_sd)):
        plt.errorbar(np.arange(8)+1, avg_train_loss[ni], yerr=std_train_loss[ni], label='Train loss')
        plt.errorbar(np.arange(8)+1, avg_eval_loss[ni], yerr=std_eval_loss[ni], label='Validation loss')
        plt.errorbar(np.arange(8)+1, avg_test_loss[ni], yerr=std_test_loss[ni], label='Test loss')
        plt.title('NN loss depending on 2nd layer unit number, noise std: ' + str(noise_sd[ni]))
        plt.ylabel('Loss')
        plt.xlabel('Number of hidden nodes in the 2nd hidden layer')
        plt.legend(loc='upper right')
        plt.show()


def plot_hist(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes):
    reg_strengths = [0.001, 0.00001]
    for i in range(len(reg_strengths)):
        model, _ = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes, regul_strength=reg_strengths[i])
        weights = model.get_weights()
        v, w = weights[0], weights[2]
        plt.title('Weights histogram for regularisation strength = ' + str(reg_strengths[i]))
        plt.xlim(-2, 2)
        plt.hist(np.concatenate((v.flatten(), w.flatten())), bins=np.arange(-2.0, 2.2, 0.2) if i == 0 else 'auto')
        plt.show()


def main():
    # compare_pcns()
    stats_reg()
    # stats_hnodes_sec_hlayers()

    # # Inputs generation
    # nb_data = 1200
    # t = np.arange(301, 1501, 1)
    # input_patterns = np.zeros((6, nb_data))
    # for i in range(nb_data):
    #     input_patterns[0][i] += x(t[i]-20)
    #     input_patterns[1][i] += x(t[i]-15)
    #     input_patterns[2][i] += x(t[i]-10)
    #     input_patterns[3][i] += x(t[i]-5)
    #     input_patterns[4][i] += x(t[i])
    #     input_patterns[5][i] = x(t[i]+5) # Targets
    # input_patterns, targets = np.split(input_patterns, [5])
    # # Split the inputs
    # data_patterns, data_targets, test_patterns, test_targets = split_inputs(input_patterns, targets.flatten(), 0.17)
    # # Build and train the network
    # nb_hidden_layers = 2
    # nb_hidden_nodes = [5, 6]
    # model, _ = train(data_patterns, data_targets, nb_hidden_layers, nb_hidden_nodes)

    # # Plot predictions vs actual time series
    # predictions = model.predict(np.transpose(input_patterns))
    # predictions = predictions.reshape(len(predictions))
    # plt.plot(t[5:1199], xs[306: 1500])
    # plt.plot(t[5:1199], predictions[:1194])
    # plt.legend(['time series', 'approximation'])
    # plt.show()


def train(patterns, targets, nb_hidden_layers, nb_hidden_nodes, regul_strength=0.00001, do_plot=False):
    """
    Train and return a two layers perceptron model.

    Params:
        patterns: input patterns
        targets: target values
        nb_hidden_layers: number of hidden layers
        nb_hidden_nodes: number of nodes (units) in the hidden layer
    
    Returns:
        The newly built and trained keras model
    """
    model = build_network(nb_hidden_layers, nb_hidden_nodes, regul_strength=regul_strength)
    # Train the network
    history = model.fit(np.transpose(patterns), np.transpose(targets), verbose=0, epochs=1000, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)])

    if do_plot:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
    return model, history


def build_network(nb_hidden_layers, nb_hidden_nodes, regul_strength=0.00001):
    """
    Build and returns a two or three layer perceptron model

    :param nb_hidden_layers:
    :param nb_hidden_nodes: a list where each element is the number of hidden nodes indexed by the number of the hidden layer
    :param regul_strength: regularization strength
    :return:
    """
    model = Sequential()
    # Add the hidden layers
    for i in range(nb_hidden_layers):
        if i == 0:
            # first hidden layer takes the inputs
            model.add(Dense(nb_hidden_nodes[i],
                            input_shape=(5,),
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))
        else:
            model.add(Dense(nb_hidden_nodes[i],
                            use_bias=True,
                            activation='sigmoid',
                            kernel_regularizer=regularizers.l1(regul_strength)))

    # Add the output layer
    model.add(Dense(1, activation='linear'))
    # Compile the model
    model.compile(optimizer=SGD(lr=0.01, momentum=0.9, nesterov=False), loss='mean_squared_error', metrics=['accuracy'])
    return model


def train_stats(nb_hidden_nodes, reg, show_preds=False):
    nb_iter = 1
    # Inputs generation
    nb_data = 1200
    t = np.arange(301, 1501, 1)
    input_patterns = np.zeros((5, nb_data))
    targets = np.zeros((nb_data))
    for i in range(nb_data):
        input_patterns[0][i] = x(t[i]-20)
        input_patterns[1][i] = x(t[i]-15)
        input_patterns[2][i] = x(t[i]-10)
        input_patterns[3][i] = x(t[i]-5)
        input_patterns[4][i] = x(t[i])
        targets[i] = x(t[i]+5)  # Targets

    # Split the inputs
    data_patterns, data_targets, test_patterns, test_targets = split_inputs(input_patterns, targets, 0.17)

    # Build and train the network
    nb_hidden_layers = 1
    # nb_hidden_nodes = 1

    avg_loss = np.zeros(nb_iter)
    avg_val = np.zeros(nb_iter)
    avg_test = np.zeros(nb_iter)
    for i in range(nb_iter):
        model = build_network(nb_hidden_layers, [nb_hidden_nodes], reg)
        # Train the network
        history = model.fit(np.transpose(data_patterns),
                            np.transpose(data_targets),
                            verbose=0,
                            epochs=1000,
                            validation_split=0.8,
                            callbacks=[
                                EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto',
                                              baseline=None, restore_best_weights=False)])
        avg_loss[i] = history.history['loss'][-1]
        avg_val[i] = history.history['val_loss'][-1]
        avg_test[i] = model.evaluate(np.transpose(test_patterns), test_targets.reshape(204, 1), verbose=0)[0]
        if show_preds is True:
            predictions = model.predict(np.transpose(input_patterns))
            predictions = predictions.reshape(len(predictions))
            plt.plot(t[5:1199], xs[306: 1500])
            plt.plot(t[5:1199], predictions[:1194])
            plt.legend(['time series', 'approximation'])
            plt.title('Time series approximation for l1 regularizer=' + str(reg))
            plt.show()

        train_mean = np.mean(avg_loss)
    train_sd = np.std(avg_loss)
    val_mean = np.mean(avg_val)
    val_sd = np.std(avg_val)
    test_mean = np.mean(avg_test)
    test_sd = np.std(avg_test)

    return train_mean, train_sd, val_mean, val_sd, test_mean, test_sd


def plot_error_nodes():
    nb_hidden_nodes = np.arange(1, 9, 1)
    train_loss = np.array([.03186934491497787,
                           .014009344215088305,
                           .012117468532474345,
                           .010911208364279501,
                           .012618332538267028,
                           .012594629281046026,
                           .012517072256635781,
                           .01297381431889257])
    train_sd = np.array([.029572793303710494,
                         .0031956480024717035,
                         .0026622021008729083,
                         .0020477448872463304,
                         .0012022175063384445,
                         .002181386130408439,
                         .0012471810505214032,
                         .0007278988919208239])
    val_loss = np.array([.030435645699716046,
                         .01741614184836615,
                         .014197436043550931,
                         .013704759344093997,
                         .014399912420524708,
                         .015005076474041552,
                         .014275850241225551,
                         .014985283180005577])
    val_sd = np.array([.025643974833894228,
                       .0032729151950947936,
                       .002397270138052556,
                       .0017784509550586474,
                       .0011750467128106043,
                       .002218485932311985,
                       .0012441565515153065,
                       .000983019792110351])
    test_loss = np.array([.030130522683555,
                          .01792131010716891,
                          .015778483905117303,
                          .014605055102055852,
                          .014330532085880929,
                          .013967466989860816,
                          .013717313954497084,
                          .01367857258255575])
    test_sd = np.array([.02400232207749552,
                        .003602178267008385,
                        .0030665944726024014,
                        .002285322309806971,
                        .001225536507615894,
                        .0022959123021655996,
                        .001255843335102543,
                        .0009005921101145326])

    plt.errorbar(nb_hidden_nodes, train_loss, 2 * train_sd)
    plt.errorbar(nb_hidden_nodes, val_loss, 2 * val_sd)
    plt.errorbar(nb_hidden_nodes, test_loss, 2 * test_sd)
    plt.legend(['Train', 'Validation', 'Test'])
    plt.title('Train, validation and test loss depending on hidden nodes')
    plt.show()


def plot_reg():
    train_mean = np.array([0.01307931,
                           0.01268448,
                           0.01336886,
                           0.0201559,
                           0.03016634,
                           0.03461447,
                           0.09025964,
                           0.09878054])
    train_sd = np.array([0.00262002,
                         0.00142763,
                         0.00252644,
                         0.00120224,
                         0.00157821,
                         0.00161525,
                         0.00037483,
                         0.00060148])
    val_mean = np.array([0.0152548,
                         0.0154016,
                         0.01607853,
                         0.02133256,
                         0.03022142,
                         0.03466484,
                         0.07945064,
                         0.08295708])
    val_sd = np.array([0.00263152,
                       0.00162979,
                       0.00276201,
                       0.00123131,
                       0.00142177,
                       0.00165389,
                       0.00050899,
                       0.00051901])
    test_mean = np.array([0.01277346,
                          0.01439313,
                          0.01636736,
                          0.02344366,
                          0.03075579,
                          0.03373647,
                          0.09336049,
                          0.08585291])
    test_sd = np.array([0.00234305,
                        0.00172069,
                        0.00258065,
                        0.00145807,
                        0.0018088,
                        0.00172831,
                        0.0003506,
                        0.00063077])


def evaluate_regularizer():
    """regularizer part"""
    regs = np.array([0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1])
    train_mean = np.zeros(len(regs))
    train_sd = np.zeros(len(regs))
    val_mean = np.zeros(len(regs))
    val_sd = np.zeros(len(regs))
    test_mean = np.zeros(len(regs))
    test_sd = np.zeros(len(regs))
    # regs = np.arange(0.001, 0.011, 0.001)
    for i, reg in enumerate(regs):
        print(i)
        train_mean[i], train_sd[i], val_mean[i], val_sd[i], test_mean[i], test_sd[i] = train_stats(5, reg,
                                                                                                   show_preds=True)
    print('train mean', train_mean, 'sd', train_sd)
    print('val mean', val_mean, 'sd', val_sd)
    print('test mean', test_mean, 'sd', test_sd)
    plt.errorbar(regs, train_mean, 2 * train_sd)
    plt.errorbar(regs, val_mean, 2 * val_sd)
    plt.errorbar(regs, test_mean, 2 * test_sd)
    plt.legend(['Train', 'Validation', 'Test'])
    plt.title('Train, validation and test loss depending on regularizer')
    plt.show()


if __name__ == "__main__":
    evaluate_regularizer()

    """non reg part (for nb of inputs)"""
    #for i in range(1, 9):
    #    print("nodes", i)
    #    train_stats(i, 0)
    #plot_error_nodes()
