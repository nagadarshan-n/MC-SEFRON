import numpy as np

class EncodingNeurons(object):
    def __init__(self, centre, width, RF):
        self.centre = centre
        self.width = width
        self.RF = RF

def generate_population(param):
    RF = param.RF
    dim = param.dim
    overlap = param.overlap
    Imin = 0
    Imax = 1

    temp = (Imin + (2*np.arange(1, RF+1)-3) / 2*(Imax-Imin)/(RF-2))
    centre = np.tile(temp, (1, dim))
    temp = []
    width = np.zeros((1, dim*RF)) + (1/overlap*(Imax-Imin)/(RF-2))

    return EncodingNeurons(centre, width, RF)
