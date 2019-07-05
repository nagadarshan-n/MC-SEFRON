from spike_response import spike_response
from STDP_norm import STDP_norm
import numpy as np

class Sample(object):
    def __init__(self, Spike_Time, Esto, U_TID, clas):
        self.Spike_Time = Spike_Time
        self.Esto = Esto
        self.U_TID = U_TID
        self.clas = clas


def Spike_code(Spike_Train, clas, param):
    Spike_Time=Spike_Train;

    Esto=spike_response(param.TID - Spike_Train.transpose(), param.tau)

    U_TID = STDP_norm((param.TID - Spike_Train), param.stdp)
    U_TID = U_TID.transpose()
    clas = int(clas-1)
    return Sample(Spike_Time, Esto, U_TID, clas)
