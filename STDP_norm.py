import numpy as np
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

def STDP_norm(r_time, tau):
    stdp = np.exp(-np.abs(r_time)/tau)

    g = deepcopy(r_time)
    g[g>0] = 1
    g[g<0] = 0
    g = g*stdp

    if (isinstance(np.sum(g, axis=0), np.ndarray)):
        pos_weight = g/((np.sum(g, axis=0))[np.newaxis, :])
    else:
        pos_weight = g/np.sum(g)

    temp = np.isnan(pos_weight)
    pos_weight[temp]=0

    g= deepcopy(r_time)
    g[g>0] = 0
    g[g<0] = 1
    g = g * stdp

    if (isinstance(np.sum(g, axis=0), np.ndarray)):
        neg_weight = -1 * (g/((np.sum(g, axis=0))[np.newaxis, :]))
    else:
        neg_weight = -1 * (g/np.sum(g))

    temp = np.isnan(neg_weight)
    neg_weight[temp] = 0

    Ut = pos_weight + neg_weight

    return Ut
