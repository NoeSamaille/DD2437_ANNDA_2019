from util import *
from two_rbms import TwoRBMs


def main():
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    print("\nStarting a Two-RBMs-DBN..")

    dbn = TwoRBMs(sizes={"vis": image_size[0] * image_size[1], "hid": 500, "top": 500},
                  image_size=image_size,
                  batch_size=20
                  )

    dbn.train_greedylayerwise(vis_trainset=train_imgs, n_iterations=10, verbose=True)

    for im in range(10):
        dbn.recall(test_imgs[im], "generations/recall_two" + str(im))
        save_img("generations/data_two" + str(im), test_imgs[im])

    # get_activations(test_imgs, rbm)

    dbn.reconstruct_error(test_imgs)


def save_pic(image, filename):
    image = image * 255
    image = Image.fromarray(image.astype(np.uint8))
    image.save(filename + '.png', format='PNG')


def get_activations(test_imgs, rbm):
    inputs = test_imgs[:rbm.batch_size]
    activations = rbm.trace_activations(inputs)
    save_pic(activations, "activations")


if __name__ == "__main__":
    main()