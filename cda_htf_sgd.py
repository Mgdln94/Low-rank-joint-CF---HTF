import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
from utils import *
import random
from pytorch_complex_tensor import ComplexTensor
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.integrate import trapz, simps


def get_basis_tst(x, K):
    M, N = x.shape
    basis = []
    for n in range(N):
        basis.append(np.zeros((K, M)))
        basis[n] = np.exp(np.outer(-1j * 2 * math.pi * np.arange((-K - 1) / 2 + 1, (K - 1) / 2 + 1), x[:, n]))
    return basis


def project_simplex(x, mask=None):
    """ Take a vector x (with possible nonnegative entries and non-normalized)
        and project it onto the unit simplex.
        mask:   do not project these entries
                project remaining entries onto lower dimensional simplex
    """
    if mask is not None:
        mask = np.asarray(mask)
        xsorted = np.sort(x[~mask])[::-1]
        # remaining entries need to sum up to 1 - sum x[mask]
        sum_ = 1.0 - np.sum(x[mask])
    else:
        xsorted = np.sort(x)[::-1]
        # entries need to sum up to 1 (unit simplex)
        sum_ = 1.0
    lambda_a = (np.cumsum(xsorted) - sum_) / np.arange(1.0, len(xsorted) + 1.0)
    for i in range(len(lambda_a) - 1):
        if lambda_a[i] >= xsorted[i + 1]:
            astar = i
            break
    else:
        astar = -1
    p = np.maximum(x - lambda_a[astar], 0)
    if mask is not None:
        p[mask] = x[mask]
    return p


def projection_simplex_Eucl(y):
    D = len(y)

    u = np.sort(y)[::-1]
    arg_u = np.argsort(y)[::-1]

    rho = 0
    for j in range(D):
        if u[j] + 1 / (j + 1) * (1 - np.sum(u[0:j + 1])) > 0:
            rho = j + 1
    lambda_val = 1.0 / rho * (1 - np.sum(u[0:rho]))

    x = np.zeros([D])
    for i in range(D):
        x[i] = max(y[i] + lambda_val, 0)
    return np.float32(x)


def train(model, optimiser, train_dataloader):
    model.train()
    for idx, data in enumerate(train_dataloader):
        # print('Init lambda')
        # print(model.factors[-1])
        # print(model.factors[-1].sum())
        x = data
        optimiser.zero_grad()
        y_hat = model(x, idx, 'train')
        loss = torch.mean(-torch.log(y_hat[y_hat > 0]))
        loss.backward()  # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True

        optimiser.step()
        # print('Before Projecting')
        # print(model.factors[-1])
        # Project on the probability simplex
        # print(torch.tensor(projection_simplex_Eucl((((model.factors[-1]).clone()).detach().numpy()))))

        # model.factors[-1] = nn.Parameter(torch.tensor(projection_simplex_Eucl((((model.factors[-1]).clone()).detach().numpy())), requires_grad=True))

        # print('After gradient step')
        # print(model.factors[-1])
        # print(model.factors[-1].sum())
        model.factors[-1].data = torch.tensor(
            projection_simplex_Eucl((((model.factors[-1]).clone()).detach().numpy())).reshape(model.F, 1))
        # print(model.factors[-1])
        # print(model.factors[-1].sum())
        # print('After Norm')
        # print('Magda')


def validate(model, val_dataloader, M):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for idx, data in enumerate(val_dataloader):
            x = data
            y_hat = model(x, idx, 'val')
            validation_loss += torch.mean(-torch.log(y_hat[y_hat > 0]))
    return validation_loss / M


class csid_net(nn.Module):
    def __init__(self, x_train, x_val, F, shape, b_size):

        super(csid_net, self).__init__()

        self.F, self.shape = F, shape  # list of K = [K1 K2 ... KN]
        self.ndims = len(self.shape)  # N

        factors_init = ComplexTensor(np.array(self.init_factors(), dtype=np.complex64))

        self.basis_train = self.get_basis(x_train)
        self.basis_val = self.get_basis(x_val)

        self.basis_train = ComplexTensor(np.array(self.basis_train, dtype=np.complex64))
        self.basis_val = ComplexTensor(np.array(self.basis_val, dtype=np.complex64))
        self.factors = nn.ParameterList(([nn.Parameter(factors_init[n]) for n in range(self.ndims)]))
        lst = []
        [lst.append(random.uniform(0, 1)) for i in range(self.F)]
        f_vec = ComplexTensor((1 / (self.F) * np.ones(self.F)).reshape(self.F, 1))
        self.factors.append(nn.Parameter(f_vec))
        self.idx_train = []

        for i in range(math.ceil(x_train.shape[0] / b_size)):
            l = i * b_size
            r = l + b_size if l + b_size < x_train.shape[0] else x_train.shape[0]
            self.idx_train.append(range(l, r))

        self.idx_val = []
        for i in range(math.ceil(x_val.shape[0] / b_size)):
            l = i * b_size
            r = l + b_size if l + b_size < x_val.shape[0] else x_val.shape[0]
            self.idx_val.append(range(l, r))

    def init_factors(self):
        factors = []
        for n in range(self.ndims):
            factors.append(np.random.rand(int((self.shape[n] - 1) / 2), self.F) + 0.1j * np.random.rand(
                int((self.shape[n] - 1) / 2), self.F))
            # factors.append(np.random.rand(int((self.shape[n]-1)/2), self.F)+0.1j * np.random.rand(int((self.shape[n]-1)/2), self.F))
        return factors

    def get_basis(self, x):
        M = x.shape[0]
        basis = []
        for n in range(self.ndims):
            basis.append(np.zeros((self.shape[0], M)))
            basis[n] = np.exp(
                np.outer(-1j * 2 * math.pi * np.arange((-self.shape[0] - 1) / 2 + 1, (self.shape[0] - 1) / 2 + 1),
                         x[:, n].numpy()))
        return basis

    def get_conditionals_per_latent(self, n):

        t = np.linspace(0.01, 1, 100)
        bb1 = ComplexTensor(np.array(np.exp(
            np.outer(-1j * 2 * math.pi * np.arange((-self.shape[0] - 1) / 2 + 1, (self.shape[0] - 1) / 2 + 1), t)),
                                     dtype=np.complex64))
        aug_fac = ComplexTensor(torch.cat((torch.cat((((ComplexTensor(self.factors[n]).real)), torch.ones(1, self.F),
                                                      (ComplexTensor(self.factors[n]).real).__reversed__()), dim=0),
                                           torch.cat((((ComplexTensor(self.factors[n]).imag)), torch.zeros(1, self.F),
                                                      (ComplexTensor(self.factors[n]).imag.neg()).__reversed__()),
                                                     dim=0)), dim=0))

        res = ((bb1.t().mm((aug_fac))).real)
        res[
            res < 0] = 0  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
        res_sc = (bb1.t().mm((aug_fac))).real
        res_sc[res_sc < 0] = 0
        res_sc = res_sc.data

        c = np.zeros(self.F)
        for col in range(res_sc.shape[1]):
            c[col] = 1 / trapz(res_sc[:, col], t)
        return c

    def forward(self, x, idx, state):
        if state == 'train':

            aug_fac = ComplexTensor(torch.cat((torch.cat((
                                                         ((ComplexTensor(self.factors[0]).real)), torch.ones(1, self.F),
                                                         (ComplexTensor(self.factors[0]).real).__reversed__()), dim=0),
                                               torch.cat((((ComplexTensor(self.factors[0]).imag)),
                                                          torch.zeros(1, self.F),
                                                          (ComplexTensor(self.factors[0]).imag.neg()).__reversed__()),
                                                         dim=0)), dim=0))

            # Ensure sum to 1 conditionals
            c = np.ones((self.ndims, self.F))
            for j in range(self.ndims):
                c[j] = self.get_conditionals_per_latent(j)

            res = ((self.basis_train[0][:, self.idx_train[idx]]).t().mm((aug_fac))).real
            res[
                res < 0] = 0  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
            res = torch.Tensor(c[0, :]) * res  # Normalize conditional density

            for n in range(1, self.ndims):
                aug_fac = ComplexTensor(torch.cat((torch.cat((((ComplexTensor(self.factors[n]).real)),
                                                              torch.ones(1, self.F),
                                                              (ComplexTensor(self.factors[n]).real).__reversed__()),
                                                             dim=0), torch.cat((((ComplexTensor(self.factors[n]).imag)),
                                                                                torch.zeros(1, self.F), (ComplexTensor(
                    self.factors[n]).imag.neg()).__reversed__()), dim=0)), dim=0))

                tmp = (((self.basis_train[n][:, self.idx_train[idx]]).t().mm((aug_fac)))).real
                tmp[tmp < 0] = 0
                tmp = torch.Tensor(c[n, :]) * tmp  # Normalize conditional density
                res = res * tmp

            # Ensure sum to 1 constraint
            fnl_results = torch.sum((res.mm(torch.diagflat(self.factors[-1]))), dim=1)
            return fnl_results

        else:
            aug_fac = ComplexTensor(torch.cat((torch.cat((
                                                         ((ComplexTensor(self.factors[0]).real)), torch.ones(1, self.F),
                                                         (ComplexTensor(self.factors[0]).real).__reversed__()), dim=0),
                                               torch.cat((((ComplexTensor(self.factors[0]).imag)),
                                                          torch.zeros(1, self.F),
                                                          (ComplexTensor(self.factors[0]).imag.neg()).__reversed__()),
                                                         dim=0)), dim=0))

            # Ensure sum to 1 conditionals
            c = np.ones((self.ndims, self.F))
            for j in range(self.ndims):
                c[j] = self.get_conditionals_per_latent(j)

            res = ((self.basis_val[0][:, self.idx_val[idx]]).t().mm((aug_fac))).real
            res[
                res < 0] = 0  # Note here: res can take negative values, take max{0,red} -> res[res < 0] = 0, Ensure positivity
            res = torch.Tensor(c[0, :]) * res  # Normalize conditional density

            for n in range(1, self.ndims):
                aug_fac = ComplexTensor(torch.cat((torch.cat((((ComplexTensor(self.factors[n]).real)),
                                                              torch.ones(1, self.F),
                                                              (ComplexTensor(self.factors[n]).real).__reversed__()),
                                                             dim=0), torch.cat((((ComplexTensor(self.factors[n]).imag)),
                                                                                torch.zeros(1, self.F), (ComplexTensor(
                    self.factors[n]).imag.neg()).__reversed__()), dim=0)), dim=0))

                tmp = (((self.basis_val[n][:, self.idx_val[idx]]).t().mm((aug_fac)))).real
                tmp[tmp < 0] = 0
                tmp = torch.Tensor(c[n, :]) * tmp  # Normalize conditional density
                res = res * tmp
            # Ensure sum to 1 constraint
            fnl_results = torch.sum((res.mm(torch.diagflat(self.factors[-1]))), dim=1)
            return fnl_results


class cda_htf_sgd:
    def __init__(self, coef_size, alpha, F, lr, b_size, max_iter=100, tol=1e-3):

        self.coef_size = coef_size
        self.alpha = alpha
        self.F = F
        self.l_rate = lr
        self.b_size = b_size
        self.max_iter = max_iter
        self.tol = tol
        self.log_val = float('inf')
        self.factors = []
        self.scale = 1 / (self.F) * np.ones(self.F)
        self.N = None
        self.max_iter_no_impr = 10  # Patience, early stopping after no improvements , 10 iterations
        self.md = None

    def fit(self, X_train, X_val):

        s_tr, _ = X_train.shape
        s_vl, self.N = X_val.shape

        X_train = torch.from_numpy(X_train).float()  # Converting numpy array to tensor
        X_val = torch.from_numpy(X_val).float()

        # `TensorDataset` provides a way to create a dataset out of the data that is already loaded into memory.
        # The idea behind the DataLoader is to load your data using multiprocessing (and pinned memory) to asynchronously
        # push your data batch onto the GPU during training so that you can basically hide the data loading time.
        # This is of course the optimal use case and if you are working with a slow HDD, you will most likely notice the data loading time.

        train_dataset = TensorDataset(X_train)  # [8505, 82]
        val_dataset = TensorDataset(X_val)

        self.md = csid_net(train_dataset.tensors[0], val_dataset.tensors[0], self.F, [self.coef_size] * self.N,
                           self.b_size)

        cost_hist = deque([float('Inf')], maxlen=self.max_iter_no_impr)
        train_dataloader = DataLoader(train_dataset, batch_size=self.b_size, shuffle=False, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=self.b_size, shuffle=False, num_workers=4)
        optimizer = optim.Adam(self.md.parameters(), lr=self.l_rate, weight_decay=self.alpha)

        cm = plt.get_cmap('gist_rainbow')
        col = [cm(1. * i / self.max_iter) for i in range(self.max_iter)]

        for i in range(self.max_iter):
            print('Iteration')
            print(i)
            train(self.md, optimizer, train_dataloader)
            print(self.md.factors[-1])
            print(self.md.factors[-1].sum())
            self.log_val = validate(self.md, val_dataloader, s_vl)

            # if i % 5 == 0:
            #     print(f'Epoch: {i} MSE: {self.log_val:.2f}')
            if i > self.max_iter_no_impr and self.log_val > max(cost_hist):
                # print(f'Epoch: {i} MSE: {self.log_val}')
                break
            cost_hist.append(self.log_val)
        self.factors = [fact.detach().numpy() for fact in self.md.factors]
