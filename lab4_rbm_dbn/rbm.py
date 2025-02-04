from util import *
import matplotlib.pyplot as plt

class RestrictedBoltzmannMachine():
    """
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    """
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep belief net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end.
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom
        if is_bottom : self.image_size = image_size
        self.is_top = is_top
        if is_top : 
            self.n_labels = n_labels

        self.batch_size = batch_size      

        self.delta_bias_v = 0
        self.delta_weight_vh = 0
        self.delta_bias_h = 0
        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))
        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))
        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        self.delta_weight_v_to_h = 0
        self.delta_weight_h_to_v = 0        
        self.weight_v_to_h = None
        self.weight_h_to_v = None

        self.learning_rate = 0.05
        self.momentum = 0.7
        self.decay = 0.0002
        self.print_period = 5000

        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 1, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        return
        
    def cd1(self, visible_trainset, labels=None, n_iterations=10000, verbose=False, disphist=False, stats_err=False):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]
        errsum = 0

        if self.is_bottom:
            viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape((self.image_size[0], self.image_size[1], -1)), it=-1,
                   grid=self.rf["grid"])

        if disphist is True:
            print("trying to plot")
            self.plot_hist(self.weight_vh, self.bias_v, self.bias_h)
        
        errors = []

        for it in range(n_iterations):

            for ind_batch in range(n_samples // self.batch_size):
                # positive phase (awake)
                batch_bounds = [ind_batch * self.batch_size, (ind_batch + 1) * self.batch_size]
                if self.is_top:
                    batch = np.concatenate((visible_trainset[batch_bounds[0]:batch_bounds[1]], labels[batch_bounds[0]:batch_bounds[1]]), axis=1)
                else:
                    batch = visible_trainset[batch_bounds[0]:batch_bounds[1]]
                pos_hprob, pos_hact = self.get_h_given_v(batch)
                
                # negative phase (asleep)
                neg_vprob, neg_vact = self.get_v_given_h(pos_hact)
                neg_hprob, neg_hact = self.get_h_given_v(neg_vact)
                
                # print error
                if verbose and (ind_batch % 1000 == 0):
                    err = np.sum((batch - neg_vact)**2)
                    print("Epoch", it, ":\n -> Recon loss", batch_bounds, ":\n  -> Square err:", err, "\n  -> Euclidian:", np.linalg.norm(batch - neg_vact))

                # updating parameters
                self.update_params(batch, pos_hact, neg_vprob, neg_hprob)

            if stats_err:
                hp, h = self.get_h_given_v(visible_trainset)
                vp, v = self.get_v_given_h(h)
                errors.append(np.sum((visible_trainset - v)**2)/n_samples)
                
            # visualize once in a while when visible layer is input images
            if it % self.rf["period"] == 0 and self.is_bottom:
                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=it, grid=self.rf["grid"])
            if disphist is True:
                print("trying to plot")
                self.plot_hist(self.weight_vh, self.bias_v, self.bias_h)
        return errors

    def update_params(self, v_0, h_0, v_k, h_k):
        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """
        pos = np.dot(np.transpose(v_0), h_0)
        pos_vb = np.sum(v_0, axis=0)
        pos_hb = np.sum(h_0, axis=0)
        neg = np.dot(np.transpose(v_k), h_k)
        neg_vb = np.sum(v_k, axis=0)
        neg_hb = np.sum(h_k, axis=0)
        self.delta_bias_v = self.momentum*self.delta_bias_v + (self.learning_rate/self.batch_size)*(pos_vb - neg_vb)
        self.bias_v += self.delta_bias_v
        self.delta_bias_h = self.momentum*self.delta_bias_h + (self.learning_rate/self.batch_size)*(pos_hb-neg_hb)                
        self.bias_h += self.delta_bias_h
        self.delta_weight_vh = self.momentum*self.delta_weight_vh + self.learning_rate*((pos - neg)/self.batch_size - self.decay*self.weight_vh)
        self.weight_vh += self.delta_weight_vh    
        return

    def get_h_given_v(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None

        h_prob = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_vh))
        h_act = sample_binary(h_prob)

        return h_prob, h_act

    def get_v_given_h(self, hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        if self.is_top:
            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:].
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            support = self.bias_v + np.dot(hidden_minibatch, np.transpose(self.weight_vh))
            data_support, label_support = support[:, :-self.n_labels], support[:, -self.n_labels:]
            data_prob = sigmoid(data_support)

            summax = np.max(label_support, axis=1)
            summax = np.reshape(summax, summax.shape+(1,)).repeat(np.shape(label_support)[1],axis=-1)
            label_support -= summax
            normalisers = np.exp(label_support).sum(axis=1)
            normalisers = np.reshape(normalisers, normalisers.shape+(1,)).repeat(np.shape(label_support)[1],axis=-1)
            label_prob = np.exp(label_support)/normalisers

            v_prob = np.concatenate((data_prob, label_prob), axis=1)
            v_act = np.concatenate((sample_binary(data_prob), label_prob), axis=1)
        else:
            v_prob = sigmoid(self.bias_v + np.dot(hidden_minibatch, np.transpose(self.weight_vh)))
            v_act = sample_binary(v_prob)

        return v_prob, v_act

    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None
        h_prob = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_v_to_h))
        h_act = sample_binary(h_prob)

        return h_prob, h_act

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None

        v_prob = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_h_to_v))
        v_act = sample_binary(v_prob)

        return v_prob, v_act

    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        batch_size = np.shape(trgs)[0]

        self.delta_weight_h_to_v = self.learning_rate / batch_size * np.transpose(trgs) @ (inps - preds)
        self.delta_bias_v = self.learning_rate * np.mean(inps - preds)
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """
        batch_size = np.shape(trgs)[0]
        self.delta_weight_v_to_h = self.learning_rate/batch_size * np.transpose(trgs) @ (inps - preds)
        self.delta_bias_h = self.learning_rate * np.mean(inps - preds)

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    

    def recall(self, image, filename):
        pos_hprob, pos_hact = self.get_h_given_v(image)
        neg_vprob, neg_vact = self.get_v_given_h(pos_hact)
        save_img(filename, neg_vprob)

    def reconstruct_error(self, vis_testset):
        _, pos_hact = self.get_h_given_v(vis_testset)
        neg_vprob, _ = self.get_v_given_h(pos_hact)
        err = np.sum((vis_testset - neg_vprob) ** 2)
        print("Reconstruction error on test set is", err)

    def trace_activations(self, batch):
        pos_hprob, _ = self.get_h_given_v(batch)
        return pos_hprob

    def loadfromfile_rbm(self, loc, name):
        self.weight_vh = np.load("%s/rbm.%s.weight_vh.npy" % (loc, name), allow_pickle=True)
        self.bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name), allow_pickle=True)
        self.bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name), allow_pickle=True)
        print("loaded rbm[%s] from %s" % (name, loc))
        return

    def savetofile_rbm(self, loc, name):
        np.save("%s/rbm.%s.weight_vh" % (loc, name), self.weight_vh)
        np.save("%s/rbm.%s.bias_v" % (loc, name), self.bias_v)
        np.save("%s/rbm.%s.bias_h" % (loc, name), self.bias_h)
        return

    def plot_hist(self, weight_vh, bias_v, bias_h):
        n, m = np.shape(weight_vh)
        weight_vh = weight_vh.reshape(n * m)
        # counts, bins = np.histogram(weight_vh)
        plt.hist(weight_vh)
        plt.title("Histogram of weight matrix")
        plt.show()

        plt.hist(bias_v)
        plt.title("Histogram of visible units' bias")
        plt.show()

        plt.hist(bias_h)
        plt.title("Histogram of hidden units' bias")
        plt.show()