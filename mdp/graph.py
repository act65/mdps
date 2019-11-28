import jax.numpy as np
import numpy
import numpy.random as rnd
from jax import grad, jit
from numpy import linalg
import mdp.utils as utils
import mdp.search_spaces as search_spaces

"""
Related??? https://arxiv.org/pdf/1901.11530.pdf
"""

def construct_mdp_basis(det_pis, mdp):
    V_det_pis = [utils.value_functional(mdp.P, mdp.r, pi, mdp.discount) for pi in det_pis]
    return np.hstack(V_det_pis)  # [n_states x n_dep_pis]

def mdp_topology(det_pis):
    # pi topology
    n = len(det_pis)
    A = numpy.zeros((n, n))
    det_pis = np.stack(det_pis)  # n x |S| x |A|
    diffs = np.sum(np.abs(det_pis[:, None, :, :]- det_pis[None, :, :, :]), axis=[2,3])
    A = (diffs == 2).astype(np.float32)
    return A

def estimate_coeffs(basis, x):
    """
    \sum \alpha_i . V^d_i = V_pi
    V_ds . a = V_pi
    Ax = b
    """
    # TODO could instead do some sort of sparse solver?
    # or min of l2 distances?
    # and they should be all positive?!
    alphas = np.dot(x, linalg.pinv(basis))
    return alphas

def mse(x, y):
    return np.sum(np.square(x-y))

def sparse_coeffs(basis, b, gamma=1e-6, lr=1e-1, a_init=None):
    """
    Want x s.t. b ~= basis . x
    min || basis . x - b ||_2^2 + gamma * ||x||_1
    """
    assert basis.shape[0] == b.shape[0]

    def sparse_loss(x):
        a = utils.softmax(x)  # convex combination
        return mse(np.dot(basis, a), b) + gamma * utils.entropy(a)

    dLdx = grad(sparse_loss)
    @jit
    def update_fn(x):
        g = dLdx(x)
        # print(x)
        return x - lr * g

    if a_init is None:
        init = 1e-3*rnd.standard_normal((basis.shape[1], ))
    else:
        init = a_init

    init = (init, np.zeros_like(init))
    output = utils.solve(search_spaces.momentum_bundler(update_fn, 0.9), init)
    a_s, mom_s = zip(*output)
    return a_s[-1]
