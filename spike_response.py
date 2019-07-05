import numpy as np

def spike_response(s, tau):
    x = s/tau
    LIF_norm = x * np.exp(1-x)
    LIF_norm[LIF_norm<0]=0

    return LIF_norm
