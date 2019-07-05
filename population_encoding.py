import numpy as np

def population_encoding(Data, Encoder, param):
    spiketime = np.zeros((Data.shape[0], Data.shape[1]*Encoder.RF))
    for j in range(Data.shape[0]):
        target = (np.tile(Data[j, :], (Encoder.RF, 1))).ravel('F')
        Firing_Strength = np.exp(-(np.power(target - Encoder.centre, 2)) / (2*np.power(Encoder.width, 2)))
        spiketime[j, :] = np.round(param.T*(1-Firing_Strength))+1;

    return spiketime
