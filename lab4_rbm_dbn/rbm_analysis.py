from util import *
from rbm import RestrictedBoltzmannMachine

def units_stats():
    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    units = np.array([500, 400, 300, 200])
    stats = []
    for i in range(len(units)):
        rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden=units[i],
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
        )
        stats.append(rbm.cd1(visible_trainset=train_imgs, n_iterations=5, verbose=True, stats_err=True))
    
    # Plot
    x = np.arange(len(stats[0]))
    for i in range(len(stats)):
        plt.plot(x, stats[i], label="h_dim="+str(units[i]))
    plt.title("Reconstruction error trough the contrastive divergence algorithm")
    plt.xlabel("Reconstruction Error")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.show()

def main():
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''

    print("\nStarting a Restricted Boltzmann Machine..")

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                     ndim_hidden=200,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
                                     )

    # try:
    #     rbm.loadfromfile_rbm(loc="trained_rbm", name="vis--hid")
    # except IOError:
    #     rbm.cd1(visible_trainset=train_imgs, n_iterations=1)
    #     rbm.savetofile_rbm("trained_single_rbm", "single")

    rbm.cd1(visible_trainset=train_imgs, n_iterations=10, disphist=False)

    for im in range(10):
        rbm.recall(test_imgs[im], "generations/recall" + str(im))
        save_img("generations/data" + str(im), test_imgs[im])

    # get_activations(test_imgs, rbm)

    rbm.reconstruct_error(test_imgs)

def save_pic(image, filename):
    image = image * 255
    image = Image.fromarray(image.astype(np.uint8))
    image.save(filename + '.png', format='PNG')

def get_activations(test_imgs, rbm):
    inputs = test_imgs[:rbm.batch_size]
    activations = rbm.trace_activations(inputs)
    save_pic(activations, "activations")

if __name__ == "__main__":
    units_stats()