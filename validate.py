import numpy as np
from FiringTime import FiringTime
from Spike_code import Spike_code
import os

def validate(Spike_data, clas, Trained_model):
    Output_class = np.zeros(clas.shape, dtype=np.int32) -1

    for j in range(Spike_data.shape[0]):
        Sample = Spike_code(Spike_data[j, :], clas[j], Trained_model)
        tc, Fire_vk = FiringTime(Trained_model.Output_neuron, Sample, Trained_model)
        firing_index = np.where(tc!=Trained_model.TOID-1)[0]

        val = np.where(np.amin(tc)==tc)[0]
        val = val[np.newaxis, :]
        vd1 = np.amin(tc)

        if (vd1 != Trained_model.TOID-1 and val.shape[0]==1):
            Output_class[j] = int(np.argmin(tc))
        elif (val.shape[0] != 1 and vd1 != Trained_model.TOID-1):
            reqd_indices = np.array([[x, y] for x in np.ravel(val, order='F') for y in np.arange(vd1-1, vd1+1)])
            fire_precision = Fire_vk[reqd_indices[:, 0], reqd_indices[:, 1]]
            fire_precision = fire_precision.reshape((np.prod(val.shape), int(fire_precision.shape/np.prod(val.shape))))
            ds = fire_precision - Trained_model.Output_neuron.theta[val]
            temp_ds = (np.zeros(val.shape) - ds[:, 1]) / (ds[:, 2] - ds[:, 1])
            f = np.argmin(temp_ds)
            Output_class[j] = val[f]

    clas = np.array(clas, dtype=np.int) - 1
    Output_class = np.array (Output_class, dtype=np.int32)
    correct = np.where(Output_class == clas)[0]
    accuracy = correct.shape[0] / Spike_data.shape[0] * 100
    return accuracy
