import numpy as np
from utils import *
import cda_htf_sgd
import itertools
import torch
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from utils import *
import matplotlib.ticker as plticker
from pytorch_complex_tensor import ComplexTensor
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import kde
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import NullFormatter

import scipy.stats as stats
from scipy.integrate import trapz, simps

# Get conditional densities in each dimension d
def get_conditionals_per_latent(md,n, h, t):

    bb1 = ComplexTensor(np.array(np.exp(np.outer(-1j * 2 * math.pi * np.arange((-md.coef_size - 1) / 2 + 1, (md.coef_size - 1) / 2 + 1), t)),dtype=np.complex64))
    aug_fac = ComplexTensor(torch.cat((torch.cat((((ComplexTensor(md.factors[n]).real)), torch.ones(1,md.F),(ComplexTensor(md.factors[n]).real).__reversed__()),dim=0),torch.cat((((ComplexTensor(md.factors[n]).imag)), torch.zeros(1,md.F),(ComplexTensor(md.factors[n]).imag.neg()).__reversed__()),dim=0)),dim=0))

    res = ((bb1.t().mm((aug_fac))).real)
    res[res < 0] = 0  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
    res_sc = (bb1.t().mm((aug_fac))).real
    res_sc[res_sc < 0] = 0
    res_sc = res_sc.data
    for col in range(res_sc.shape[1]):
        c = 1 / trapz(res_sc[:, col],t)
        res.data[:, col] = c * (res.data[:, col])  # Normalize conditional density, Ensure integration to 1
    res = res.detach().numpy()
    return res[:,h]

# Rejection sampling
def rejection_sampler(pdf, t, num_samples=1, xmin=0, xmax=1): # https://gist.github.com/rsnemmen/d1c4322d2bc3d6e36be8
    pmin = 0.
    pmax = pdf.max()
    # Counters
    naccept = 0
    ntrial = 0
    # Keeps generating numbers until we achieve the desired n
    ran = []  # output list of random numbers
    while naccept < num_samples:
        x = np.random.uniform(xmin, xmax)  # x'
        x_indx = (np.abs(t - x)).argmin()
        y = np.random.uniform(pmin, pmax)  # y'
        if y < pdf[x_indx]:
            ran.append(x)
            naccept = naccept + 1
        ntrial = ntrial + 1
    return np.asarray(ran)


def predict(md, x, basis):
    M, N = x.shape
    K, F = md.factors[0].shape
    res = torch.Tensor(np.ones((M, F)))

    basis = ComplexTensor(np.array(basis, dtype=np.complex64))
    for n in range(N):
        aug_fac = ComplexTensor(torch.cat((torch.cat((((ComplexTensor(md.factors[n]).real)), torch.ones(1,F), (ComplexTensor(md.factors[n]).real).__reversed__()), dim=0), torch.cat((((ComplexTensor(md.factors[n]).imag)), torch.zeros(1,F), (ComplexTensor(md.factors[n]).imag.neg()).__reversed__()), dim=0)), dim=0))
        tmp = (((basis[n][:,:]).t().mm((aug_fac)))).real
        tmp[tmp < 0] = 0
        res = res * tmp
    # Ensure sum to 1 constraint
    fnl_results = torch.sum((res.mm(torch.diagflat(torch.Tensor(md.factors[-1])))), dim=1)
    return fnl_results

def sample(md, num_samples, t):
    rand_lamda_index =  np.random.choice(md.F, num_samples, replace=True, p= np.squeeze((md.factors[-1])/(md.factors[-1]).sum()))
    samples = np.ones((num_samples, md.N))
    for i in range(num_samples):
        for n in range(md.N): #rejection_sampler(get_conditionals_per_latent(md,n,t)[:,h], t, num_samples)
            samples[i,n] = rejection_sampler(get_conditionals_per_latent(md,n,rand_lamda_index[i],t), t)
            #im = Image.fromarray(samples[i,n].resize(16,16))
            #im.show()
    return samples

def cda_regression_sgd(X, X_test, n_coef, alpha, F, lr, fold, b_size=512, max_iter=1000, tol=1e-2):
    print('Running cda-HTF SGD')
    MSE_folds_val = np.zeros((len(n_coef)*len(alpha)*len(F)*len(lr), fold))
    MSE_folds_test = np.zeros((len(n_coef)*len(alpha)*len(F)*len(lr), fold))

    # parameter_comb -> list[(K,a,F,lr),...,()]
    parameter_comb = list(itertools.product(*[n_coef, alpha, F, lr]))
    kf = KFold(n_splits=fold)

    n_split = 0
    for train_index, val_index in kf.split(X):
        print(f'Number of splits: {n_split}')
        X_train= X[train_index]
        X_val= X[val_index]

        for it, param in enumerate(parameter_comb):
            coef_s, a, f, lr = param
            md = cda_htf_sgd.cda_htf_sgd(coef_s, a, f, lr, b_size, max_iter, tol) # cda_htf_sgd -> call the constructor of the class
            md.fit(X_train, X_val)
            MSE_folds_val[it, n_split] = md.log_val

        n_split += 1

    # Choose the model based on the average of the VALIDATION set
    idx1 = np.argmax(np.mean(MSE_folds_val, 1))
    idx2 = np.argmax(MSE_folds_val[idx1, :])
    print('Chosen parameters')
    print(parameter_comb[idx1])

    # Retrain with all the training samples
    coef_s, a, f, lr = parameter_comb[idx1]
    md = cda_htf_sgd.cda_htf_sgd(coef_s, a, f, lr, b_size, max_iter, tol)
    md.fit(X, X)

    stp = 0.01
    num_samples = 1000
    # Sample from the distribution --> Latent variable NB H->X
    t = np.arange(0, 1, stp)
    X_samples = sample(md, num_samples, t)

    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    loc = plticker.MultipleLocator(base=0.05)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    #ax.get_legend().remove()
    plt.grid()
    ax.set_axisbelow(True)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    loc = plticker.MultipleLocator(base=0.05)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    #ax.get_legend().remove()
    plt.grid()
    ax.set_axisbelow(True)
    cmap = sns.light_palette("#2ecc71", as_cmap=True)
    plt.scatter(X_samples[:, 0], X_samples[:, 1])

    # Save the test score
    basis_test = cda_htf_sgd.get_basis_tst(X_test, md.coef_size)
    target_est_test = predict(md, X_test, basis_test)
    #MSE_folds_test[it, n_split] = np.linalg.norm(target_est_test - Y_test) ** 2 / X_test.shape[0]
    print(f'coef: {coef_s}, regul:{a}, rank:{f}, lr:{lr}, b_size:{b_size}, MSE_test: {MSE_folds_test[it, n_split]}')

    print('Magda')

    return MSE_folds_test[idx1, idx2] # Return the test error
