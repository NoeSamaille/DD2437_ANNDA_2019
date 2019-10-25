from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_gibbs_recog = 15
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 20
        self.print_period = 2000
        return

    def recognize(self, true_img, true_lbl, plot=False):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        vis = true_img # visible layer gets the image data
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels

        predicted_lbl = np.zeros(true_lbl.shape)
        # for i in range(self.n_gibbs_recog):
        #     print("Gibbs sample", i)
        _, h = self.rbm_stack['vis--hid'].get_h_given_v_dir(vis)
        _, h = self.rbm_stack['hid--pen'].get_h_given_v_dir(h)
        _, h = self.rbm_stack['pen+lbl--top'].get_h_given_v(np.concatenate((h, lbl), axis=1))
        for i in range(self.n_gibbs_recog):
            _, h = self.rbm_stack['pen+lbl--top'].get_v_given_h(h)
            _, h = self.rbm_stack['pen+lbl--top'].get_h_given_v(h)
        _, v = self.rbm_stack['pen+lbl--top'].get_v_given_h(h)
        predicted_lbl += v[:,self.sizes["pen"]:]

        if plot:
            wrong = 0
            i = 0
            pred = np.argmax(predicted_lbl,axis=1)
            true = np.argmax(true_lbl,axis=1)
            while wrong < 50 and i < n_samples:
                if pred[i] != true[i]:
                    save_img(str(wrong)+"_"+str(pred[i]), true_img[i])
                    wrong += 1
                i += 1
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        return

    def generate(self, true_lbl, name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_sample = true_lbl.shape[0]

        records = []
        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        # vis = np.random.choice(2, (1, self.sizes["pen"])) * 10
        vis = np.ones((1, self.sizes["pen"])) * 10
        _, h = self.rbm_stack['pen+lbl--top'].get_h_given_v(np.concatenate((vis, lbl), axis=1))
        for _ in range(self.n_gibbs_gener):
            _, h = self.rbm_stack['pen+lbl--top'].get_v_given_h(h)
            h = np.concatenate((h[:, :self.sizes["pen"]], lbl), axis=1)
            h, _ = self.rbm_stack['pen+lbl--top'].get_h_given_v(h)

        for _ in range(50):
            h_s = np.copy(h)
            _, h_s = self.rbm_stack['pen+lbl--top'].get_v_given_h(h_s)
            _, h_s = self.rbm_stack['hid--pen'].get_v_given_h_dir(h_s[:, :self.sizes["pen"]])
            vprob, vis = self.rbm_stack['vis--hid'].get_v_given_h_dir(h_s)
            records.append([ax.imshow(vprob.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None)])

        # Set up formatting for the movie files
        import matplotlib.animation as animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

        stitch_video(fig,records).save("animations/%s.generate%d.mp4"%(name,np.argmax(true_lbl)), writer=writer)

        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations, verbose=False):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")

        except IOError :
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """
            inputs = vis_trainset
            probs_in = vis_trainset
            self.rbm_stack['vis--hid'].cd1(inputs, n_iterations=n_iterations, verbose=verbose)
            probs_in, inputs = self.rbm_stack['vis--hid'].get_h_given_v(probs_in)
            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack['vis--hid'].untwine_weights()

            print ("training hid--pen")
            """ 
            CD-1 training for hid--pen 
            """            
            self.rbm_stack['hid--pen'].cd1(inputs, n_iterations=n_iterations, verbose=verbose)
            probs_in, inputs = self.rbm_stack['hid--pen'].get_h_given_v(probs_in)
            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack['hid--pen'].untwine_weights()

            print ("training pen+lbl--top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            self.rbm_stack['pen+lbl--top'].cd1(inputs, labels=lbl_trainset, n_iterations=n_iterations, verbose=verbose)
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")

        return

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)

        updates h given v matrices
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            n_samples = vis_trainset.shape[0]
            # self.rbm_stack['vis--hid'].untwine_weights()
            # self.rbm_stack['hid--pen'].untwine_weights()
            
            for it in range(n_iterations):
                print("iteration=%7d"%it)
                for b in range(n_samples // self.batch_size):
                    inputs = vis_trainset[b * self.batch_size:(b+1) * self.batch_size, :]
                    lbl = lbl_trainset[b * self.batch_size:(b+1) * self.batch_size]

                    """ 
                    wake-phase : drive the network bottom-to-top using visible and label data
                    """
                    _, h_vis_hid = self.rbm_stack['vis--hid'].get_h_given_v_dir(inputs)
                    _, h_hid_pen = self.rbm_stack['hid--pen'].get_h_given_v_dir(h_vis_hid)
                    h_penlbl_prob, h_penlbl_top = self.rbm_stack['pen+lbl--top'].get_h_given_v(np.concatenate((h_hid_pen, lbl), axis=1))

                    """
                    alternating Gibbs sampling in the top RBM : also store neccessary information for learning this RBM
                    """
                    h_penlbl_gs = np.copy(h_penlbl_top) # we need to keep h_penlbl_top for updates
                    for i in range(self.n_gibbs_wakesleep):
                        _, v_penlbl_top = self.rbm_stack['pen+lbl--top'].get_v_given_h(h_penlbl_gs)
                        v_penlbl_top = np.concatenate((v_penlbl_top[:, :self.sizes["pen"]], lbl), axis=1)
                        _, h_penlbl_top_gs = self.rbm_stack['pen+lbl--top'].get_h_given_v(v_penlbl_top)

                    # self.savetofile_rbm(loc="trained_dbm", name="pen+lbl--top")

                    """
                    sleep phase : from the activities in the top RBM, drive the network top-to-bottom
                    """
                    _, v_penlbl_top = self.rbm_stack['pen+lbl--top'].get_v_given_h(h_penlbl_top_gs)
                    _, v_hid_pen = self.rbm_stack['hid--pen'].get_v_given_h_dir(v_penlbl_top[:, :self.sizes["pen"]])
                    v_vis_hid_prob, v_vis_hid = self.rbm_stack['vis--hid'].get_v_given_h_dir(v_hid_pen)

                    """
                    predictions : compute generative predictions from wake-phase activations, 
                                  and recognize predictions from sleep-phase activations
                    """
                    # generative preds (using wake phase). Use probs according to Hinton alg
                    v_vis_hid_prob_gen, _ = self.rbm_stack['vis--hid'].get_v_given_h_dir(h_vis_hid)
                    v_hid_pen_gen, _ = self.rbm_stack['hid--pen'].get_v_given_h_dir(h_hid_pen)

                    # recognize preds (using sleep phase)
                    h_hid_pen_rec, _ = self.rbm_stack['hid--pen'].get_h_given_v_dir(v_hid_pen)
                    h_vis_hid_rec, _ = self.rbm_stack['vis--hid'].get_h_given_v_dir(v_vis_hid_prob)

                    """ 
                    update generative parameters :
                    here you will only use "update_generate_params" method from rbm class
                    """
                    self.rbm_stack['vis--hid'].update_generate_params(inputs, h_vis_hid, v_vis_hid_prob_gen)
                    self.rbm_stack['hid--pen'].update_generate_params(h_vis_hid, h_hid_pen, v_hid_pen_gen)

                    """ 
                    update parameters of top rbm:
                    here you will only use "update_params" method from rbm class
                    """
                    self.rbm_stack['pen+lbl--top'].update_params(np.concatenate((h_hid_pen, lbl), axis=1),
                                                                 h_penlbl_top,
                                                                 v_penlbl_top,
                                                                 h_penlbl_top_gs)

                    """ 
                    update generative parameters :
                    here you will only use "update_recognize_params" method from rbm class
                    """
                    self.rbm_stack['vis--hid'].update_recognize_params(v_hid_pen, v_vis_hid_prob, h_vis_hid_rec)
                    self.rbm_stack['hid--pen'].update_recognize_params(v_penlbl_top[:, :self.sizes["pen"]], v_hid_pen, h_hid_pen_rec)

            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name), allow_pickle=True)
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name), allow_pickle=True)
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name), allow_pickle=True)
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name), allow_pickle=True)
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name), allow_pickle=True)
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name), allow_pickle=True)
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name), allow_pickle=True)
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
