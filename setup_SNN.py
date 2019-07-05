import numpy as np

class Model(object):
    def __init__(self, RF, pre, post, desired, precision, learning_rate, efficacy_update_range, tau, epoch, stdp):
        self.RF                    = RF
        self.sigma                 = efficacy_update_range / precision
        self.l_rate                = learning_rate
        self.T                     = pre / precision
        self.TOID                  = post/precision
        self.t_train               = np.arange(self.T+1)
        self.no_epoch              = epoch
        self.TID                   = desired / precision
        self.tau                   = tau / precision
        self.overlap               = 0.7
        self.stdp                  = stdp/precision
        self.Tm                    = 0.05/precision
        self.Td                    = self.TID
        self.Output_neuron         = None
        return

    def gaussian_function(self, A, B):
        result = []
        for i in range(A.shape[0]):
            result.append(np.exp((-np.power(A[i]-B, 2)) / (2*(self.sigma**2))))
        return np.array(result)

    def add_dataset_params(self, no_class, dim, t):
        self.no_class = int(no_class)
        self.dim = dim
        self.t = t
        return

    def initialise_theta(self, lst):
        self.theta = lst

def setup_SNN(RF, pre, post, desired, precision, learning_rate, efficacy_update_range, tau, epoch, stdp):
    return Model(RF, pre, post, desired, precision, learning_rate, efficacy_update_range, tau, epoch, stdp)
