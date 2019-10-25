from util import *
from rbm import RestrictedBoltzmannMachine


class TwoRBMs():
    '''
    network          : [top] <---> [hid] ---> [vis]
    top : top
    hid : hidden
    vis : visible
    '''

    def __init__(self, sizes, image_size, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"],
                                                   ndim_hidden=sizes["hid"],
                                                   is_bottom=True,
                                                   image_size=image_size,
                                                   batch_size=batch_size),
            'hid--top': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"],
                                                   ndim_hidden=sizes["top"],
                                                   is_top=False,
                                                   batch_size=batch_size),
            # 'pen+lbl--top': RestrictedBoltzmannMachine(ndim_visible=sizes["pen"] + sizes["lbl"],
            #                                            ndim_hidden=sizes["top"],
            #                                            is_top=True,
            #                                            n_labels=n_labels,
            #                                            batch_size=batch_size)
        }
        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_gibbs_recog = 15
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 5
        self.print_period = 2000
        return

    def train_greedylayerwise(self, vis_trainset, n_iterations, verbose=False):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack.
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try:

            self.loadfromfile_rbm(loc="two_rbm", name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()

            self.loadfromfile_rbm(loc="two_rbm", name="hid--top")

        except IOError:

            print("training vis--hid")
            inputs = vis_trainset
            probs_in = vis_trainset
            self.rbm_stack['vis--hid'].cd1(inputs, n_iterations=n_iterations, verbose=verbose)
            probs_in, inputs = self.rbm_stack['vis--hid'].get_h_given_v(probs_in)
            self.savetofile_rbm(loc="two_rbm", name="vis--hid")
            self.rbm_stack['vis--hid'].untwine_weights()

            print("training hid--top")
            self.rbm_stack['hid--top'].cd1(inputs, n_iterations=n_iterations, verbose=verbose)
            self.savetofile_rbm(loc="two_rbm", name="hid--top")

        return

    def loadfromfile_rbm(self, loc, name):

        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name), allow_pickle=True)
        self.rbm_stack[name].bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name), allow_pickle=True)
        self.rbm_stack[name].bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name), allow_pickle=True)
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):

        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def loadfromfile_dbn(self, loc, name):

        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy" % (loc, name), allow_pickle=True)
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy" % (loc, name), allow_pickle=True)
        self.rbm_stack[name].bias_v = np.load("%s/dbn.%s.bias_v.npy" % (loc, name), allow_pickle=True)
        self.rbm_stack[name].bias_h = np.load("%s/dbn.%s.bias_h.npy" % (loc, name), allow_pickle=True)
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_dbn(self, loc, name):

        np.save("%s/dbn.%s.weight_v_to_h" % (loc, name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v" % (loc, name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v" % (loc, name), self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h" % (loc, name), self.rbm_stack[name].bias_h)
        return

    def recall(self, image, filename):
        _, hact_vis_hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(image)
        _, hact_hid_top = self.rbm_stack["hid--top"].get_h_given_v(hact_vis_hid)
        _, vact_hid_top = self.rbm_stack["hid--top"].get_v_given_h(hact_hid_top)
        vact_vis_hid, _ = self.rbm_stack["vis--hid"].get_v_given_h_dir(vact_hid_top)
        save_img(filename, vact_vis_hid)

    def reconstruct_error(self, vis_testset):
        _, hact_vis_hid = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_testset)
        _, hact_hid_top = self.rbm_stack["hid--top"].get_h_given_v(hact_vis_hid)
        _, vact_hid_top = self.rbm_stack["hid--top"].get_v_given_h(hact_hid_top)
        vact_vis_hid, _ = self.rbm_stack["vis--hid"].get_v_given_h_dir(vact_hid_top)
        err = np.sum((vis_testset - vact_vis_hid) ** 2)
        print("Reconstruction error on test set is", err)

