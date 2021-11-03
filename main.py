


#conda activate C:\anaconda\envs\Time_Series_Density_estimation
# Python 3.7, see requirments

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import ShuffleSplit
from utils import *
import argparse
import torch
import pandas as pd
from model_cross_validation import *

import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
plt.rcParams["axes.grid"] = False
from sklearn import mixture
import scipy.stats as stats
from scipy.integrate import trapz, simps



def main():
    n_samples = 5000

    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)[0].astype(np.float32)
    noisy_circles = datasets.make_circles(n_samples=n_samples, noise=0.05)[0].astype(np.float32)
    orig_data = noisy_circles
    scale_normalize = MinMaxScaler(feature_range=(0, 1))
    scale_normalize.fit(orig_data)
    X_data_norm = scale_normalize.transform(orig_data)


    p_test = 0.2
    n_sims, fold = 1, 2
    n_alg = 2
    ss = ShuffleSplit(n_splits=n_sims, test_size=p_test)
    md = [[] for n in range(n_alg)]
    MSE_test = np.zeros((n_sims, n_alg))

    for it, ind in enumerate(ss.split(X_data_norm)):

        print(f'Simulation number : {it}')
        ind_tr, ind_te = ind
        s_tr, s_te = len(ind_tr), len(ind_te)

        # Training phase
        X_norm_train = X_data_norm[ind_tr]
        X_norm_test = X_data_norm[ind_te]

        # Parameters!!!
        n_coef = [11]
        alpha = [1e-2] # Regularization
        F = [6]
        lr = [1e-2]
        print(cda_regression_sgd(X_norm_train, X_norm_test, n_coef, alpha, F, lr, fold, b_size=200, max_iter=50))
        print(MSE_test)

if __name__ == '__main__':
            main()