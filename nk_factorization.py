
import functools
import numpy as np
import scipy.linalg
from numpy import linalg
from nk_utils import mtkrprod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression
import time
from tensorly import kr


class NCPD:

    def __init__(self, rank, vals, mu=0, max_iter=100, tol=1e-3, inner_max_itr=10):
        self.rank = rank
        self.vals = vals
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol
        self.inner_max_itr = inner_max_itr

        self.x_norm = None
        self.ndim = None
        self.shape = None
        self.U, self.U_d, self.UtU = [], [], []

        self.x = None
        self.cost_fit_hist = []
        self.cost_l2_reg_hist = []

    def inner_prod(self, x):
        res = np.zeros(self.rank)
        for r in range(self.rank):
            tmp = x
            for n in range(self.ndim):
                tmp = np.tensordot(tmp, self.U[n][:, r], axes=(0, 0))
            res[r] = tmp
        return res.sum()

    def norm(self):
        return np.sqrt(functools.reduce(np.multiply, ([self.U[n].T @ self.U[n] for n in range(self.ndim)])).sum())

    def cost(self, x):
        cost_fit = self.x_norm ** 2 - 2 * self.inner_prod(x) + self.norm() ** 2
        cost_l2_reg = self.mu * sum([linalg.norm(self.U[n]) ** 2 for n in range(self.ndim)])
        cost_rel = np.sqrt(cost_fit) / self.x_norm
        return cost_fit, cost_l2_reg, cost_rel

    def ao_admm_sub(self, WtW, WtY, U, U_d, ab):
        rho = np.trace(WtW) / self.rank
        cholesky_l = np.linalg.cholesky(WtW + rho * np.eye(self.rank))

        for itr in range(self.inner_max_itr):
            # primal updates
            U_t = scipy.linalg.solve_triangular(cholesky_l, WtY + rho * (U + U_d).T, lower=True)
            U_t = scipy.linalg.solve_triangular(cholesky_l.T, U_t)
			
			# dual update -- Positivity
            U = (U_t.T - U_d).clip(min=0)

            # dual update -- Increasing property
            for j in range(self.rank):
                U[:,j] = IsotonicRegression().fit_transform(ab,U[:,j])

            U_d = U_d + U - U_t.T
        return U, U_d #H,U

    def fit(self, x):
        self.x_norm = linalg.norm(x)
        self.ndim = x.ndim
        self.shape = x.shape


        # Cache the tensor unfoldings
        unfolded_tensors = []
        for mode in range(self.ndim):
            unfolded_tensors.append(np.reshape(np.moveaxis(x.data, mode, 0), (x.shape[mode], -1)))

        # CPD model initialization
        self.U = [np.random.rand(self.shape[n], self.rank) for n in range(self.ndim)]
        self.U_d = [np.zeros((self.shape[n], self.rank)) for n in range(self.ndim)]
        self.UtU = [self.U[n].T @ self.U[n] for n in range(self.ndim)]

        for itr in range(self.max_iter):
            #print(f'Iteration number: {itr}')
            for n in range(self.ndim):
                UtU_mult = np.ones((self.rank, self.rank))
                for k in range(self.ndim):
                    if k != n:
                        UtU_mult = UtU_mult * self.UtU[k]

                WtW = UtU_mult + self.mu * np.eye(self.rank)
                WtY = (unfolded_tensors[n] @ kr(self.U[:n] + self.U[n + 1:])).T
				#
                ab = self.vals[n]
                self.U[n], self.U_d[n] = self.ao_admm_sub(WtW, WtY, self.U[n], self.U_d[n], ab)
                self.UtU[n] = self.U[n].T @ self.U[n]
        _, _, cost_rel = self.cost(x)
        return self.U, cost_rel