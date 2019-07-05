import numpy as np
from spike_response import spike_response
import sys
np.set_printoptions(threshold=1000)

def FiringTime(Output_neuron, Sample, param):

    W_sample = Output_neuron.weight[np.tile(np.arange(param.dim*param.RF), (1, param.no_class)),
                             np.array(np.tile(Sample.Spike_Time, (1, param.no_class)), dtype=np.int32) - 1,
                             np.ravel(np.tile(np.arange(param.no_class), (param.dim*param.RF, 1)), order='F')]


    wh = np.reshape(W_sample, (param.dim*param.RF, param.no_class), order='F')

    V = np.matmul(wh.transpose(), spike_response(param.t - Sample.Spike_Time[:, np.newaxis], param.tau))

    firing = (V>Output_neuron.theta[:, np.newaxis])

    firing_time = np.argmax(firing.transpose(), axis=0)
    firing_time[firing_time==0] = param.TOID - 1

    tc = firing_time

    return tc, V
