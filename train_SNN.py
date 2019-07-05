import numpy as np
from generate_population import generate_population
from population_encoding import population_encoding
from MC_SEFRON import MC_SEFRON

def train_SNN(model, Tr, Te):
    Train_class = Tr[:, -1]
    Train = Tr[:, :-1]
    train_size = Train.shape[0]

    Test_class = Te[:, -1]
    Test = Te[:, :-1]
    test_size = Test.shape[0]

    no_class = np.amax(Train_class)
    dim = Train.shape[1]
    t = np.matmul(np.ones((dim*model.RF, 1)), np.arange(model.TOID+1).reshape(1, int(model.TOID+1)))

    model.add_dataset_params(no_class, dim, t)

    Encoding_neurons = generate_population(model)
    Spike_Train = population_encoding(Train, Encoding_neurons, model)
    Spike_Test = population_encoding(Test, Encoding_neurons, model)

    Trained_model=MC_SEFRON(model, Spike_Train, Train_class, train_size, Spike_Test, Test_class)

    return Trained_model
