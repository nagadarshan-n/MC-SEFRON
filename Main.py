import numpy as np
import pandas as pd
from setup_SNN import setup_SNN
from train_SNN import train_SNN


def main():
    # datasets
    Tr                    = np.array(pd.read_csv('iris_train_1.csv', header=None).values)
    Te                    = np.array(pd.read_csv('iris_test_1.csv', header=None).values)

    # Algorithm parameters
    RF                    = 6
    pre                   = 3
    post                  = 4
    desired               = 2
    precision             = 0.01
    learning_rate         = 0.5
    efficacy_update_range = 0.55
    tau                   = 3
    epoch                 = 100
    stdp                  = 1.6

    model = setup_SNN(RF, pre, post, desired, precision, learning_rate, efficacy_update_range, tau, epoch, stdp)
    Trained_model = train_SNN(model, Tr, Te)
    return

if __name__ == '__main__':
    main()
