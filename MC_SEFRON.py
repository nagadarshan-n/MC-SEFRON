import numpy as np
import matplotlib.pyplot as plt
from Spike_code import Spike_code
from FiringTime import FiringTime
from copy import deepcopy
from STDP_norm import STDP_norm
from spike_response import spike_response
from validate import validate

class OutputNeuron(object):
    def __init__(self, weight, theta):
        self.weight = weight
        self.theta = theta


def MC_SEFRON(model, Spike_Train, Train_class, train_size, Spike_Test, Test_class):
    np.random.seed(0)
    no_epoch = int(model.no_epoch)
    Output_size = 0;
    model.initialise_theta([]);

    Train_accu = np.zeros(no_epoch)
    Test_accu = np.zeros(no_epoch)

    weight = np.zeros((int(model.dim*model.RF), int(model.T+1), int(model.no_class)))
    theta = np.zeros(int(model.no_class))
    Output_neuron = OutputNeuron(weight, theta)

    order = np.arange(train_size)

    # Initialize weight and firing threshold
    for j in range(train_size):
        Sample = Spike_code(Spike_Train[order[j], :], Train_class[order[j]], model)

        if (Output_neuron.theta[Sample.clas]==0):
            Output_size = Output_size + 1
            Output_neuron.weight[:, :, Sample.clas] = np.multiply((model.gaussian_function(model.t_train, Sample.Spike_Time.transpose())).transpose(), Sample.U_TID[:, np.newaxis])
            Output_neuron.theta[Sample.clas] = (np.matmul(Sample.U_TID[np.newaxis, :], Sample.Esto[:, np.newaxis])).squeeze()
        if (Output_size == model.no_class):
            break

    # Training MC_SEFRON
    for epoch in range(no_epoch):
        for j in range(train_size):
            Sample = Spike_code(Spike_Train[order[j], :], Train_class[order[j]], model)
            tc, _ = (FiringTime(Output_neuron, Sample, model))


            Other_class = np.array(list(set(np.arange(model.no_class)) - set([Sample.clas])))

            tcc = tc[Sample.clas]
            tmc = np.amin(tc[Other_class])
            reference_time = deepcopy(tc)

            if (tmc < tcc+model.Tm):

                # Determine reference post synaptic spike time
                if (tcc > model.Td-1):
                    reference_time[Sample.clas] = model.Td - 1

                trf_mc = min(model.TOID-1, tcc+model.Tm)
                Wrng_class = np.where(tc[Other_class] < tcc+model.Tm)[0]
                reference_time[Other_class[Wrng_class]] = trf_mc


                ## Weight Update
                r_time = (tc[np.newaxis, :]+1) - Sample.Spike_Time[:, np.newaxis]

                Ut = STDP_norm(r_time, model.stdp)

                r_time = (reference_time[np.newaxis, :]+1) - Sample.Spike_Time[:, np.newaxis]

                Ut_de = STDP_norm(r_time, model.stdp)

                w_tf = Output_neuron.theta / np.sum(Ut * spike_response((tc[np.newaxis, :]+1) - Sample.Spike_Time[:, np.newaxis], model.tau), axis=0)

                w_td = Output_neuron.theta / np.sum(Ut_de * spike_response((reference_time[np.newaxis, :]+1) - Sample.Spike_Time[:, np.newaxis], model.tau), axis=0)

                delta_W = np.multiply(Ut_de, (w_td - w_tf)[np.newaxis, :])

                delta_wx = np.multiply(model.gaussian_function(np.arange(model.T+1), np.tile(Sample.Spike_Time, (1, model.no_class)).squeeze()).transpose(), np.ravel(delta_W, order='F')[:, np.newaxis])

                Output_neuron.weight = Output_neuron.weight + model.l_rate * np.transpose(np.reshape(delta_wx.transpose(), (int(model.T+1), np.array(model.dim*model.RF, dtype=np.int32), model.no_class), order='F'), (1, 0, 2))
                Output_neuron.weight[Output_neuron.weight==-np.inf] = np.inf
                Output_neuron.weight[np.isnan(Output_neuron.weight)] = np.inf


        Trained_model = deepcopy(model)
        Trained_model.Output_neuron = Output_neuron
        Train_accu[epoch] = validate(Spike_Train, Train_class, Trained_model)

        Test_accu[epoch] = validate(Spike_Test, Test_class, Trained_model)

        print ("Epoch: {}/{} \nTraining Accuracy: {} \nTesting Accuracy: {}\n".format(epoch+1, no_epoch, Train_accu[epoch], Test_accu[epoch]), flush=True)



    plt.plot(Train_accu)
    plt.plot(Test_accu)
    plt.show()
    return Trained_model
