import jax.numpy as np
from jax import jit, grad
import numpy.random as rnd

import matplotlib.pyplot as plt

from mdp.utils import *
from mdp.search_spaces import *

def vi_vector_field(mdp, qs, lr):
    update_fn = value_iteration(mdp, lr)
    delta = lambda q: np.max(update_fn(q) - q, axis=1)
    return np.vstack([delta(q) for q in qs])

def pvi_vector_field(mdp, many_cores, lr):
    update_fn = parameterised_value_iteration(mdp, lr)
    delta = lambda cores: np.max(build(cores),axis=1) - np.max(build(update_fn(cores)), axis=1)
    return np.vstack([delta(cores) for cores in many_cores])

def fitted_cores(mdp, qs):
    """
    For a set of target values, want to init cores that yield these values.
    """
    cores = random_parameterised_matrix(2, 2, d_hidden=8, n_hidden=4)
    approxs = [approximate(q, cores) for q in qs]
    return approxs

def normalize(x):
    mags = np.linalg.norm(x, axis=1, keepdims=True)
    return x/mags

def plt_field(xs, dxs):
    normed_dxs = normalize(dxs)
    plt.quiver(xs[:, 0], xs[:, 1], normed_dxs[:, 0], normed_dxs[:,1], np.linalg.norm(dxs, axis=1))
    plt.colorbar()

def generate_pvi_vs_vi(mdp):
    """
    How does search in the parameter space versus the value space
    change the dynamics?
    """
    print('\nRunning PVI vs VI')
    lr = 0.1

    pis = gen_grid_policies(N=11)
    vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
    qs = [np.einsum('ijk,i->jk', mdp.P, v) for v in vs]

    plt.figure(figsize=(16,16))

    # pvi
    many_cores = fitted_cores(mdp, qs)
    dpvis = pvi_vector_field(mdp, many_cores, lr)

    plt.subplot(2,1,1)
    plt.title('Pamameterised VI')
    plt_field(vs, dpvis)

    # vi
    dvis = vi_vector_field(mdp, qs, lr)

    plt.subplot(2,1,2)
    plt.title('VI')
    plt_field(vs, dvis)

    # plt.show() # Replaced with close
    plt.close()


def cts_lr_fields(mdp):
    """
    How does the learning rate change the vector field???
    """
    n = 3
    lrs = np.logspace(-5, 0, n*n)

    pis = gen_grid_policies(N=31)
    vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
    qs = [np.einsum('ijk,i->jk', mdp.P, v) for v in vs]
    many_cores = fitted_cores(mdp, qs)

    plt.figure(figsize=(16,16))
    plt.title('PVI')
    for i, lr in enumerate(lrs):

        dpvis = pvi_vector_field(mdp, many_cores, lr)
        # dont expect vi to change with the lr?!
        # dvis = vi_vector_field(mdp, qs, lr)

        plt.subplot(n, n, i+1)
        plt.title('lr: {:.3f}'.format(lr))
        plt_field(vs, dpvis)

        # plt.title('Pamameterised VI')
    # plt.savefig('figs/lr_limit_{:.3f}.png'.format(lr))
    plt.savefig('figs/lr_limit_pvi.png', dpi=300)
    plt.close()

def lr_field_diffs(mdp):
    """
    How does the learning rate change the vector field???
    """

    pis = gen_grid_policies(N=11)
    vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
    qs = [np.einsum('ijk,i->jk', mdp.P, v) for v in vs]
    many_cores = fitted_cores(mdp, qs)

    plt.figure(figsize=(16,16))
    plt.title('PVI')

    dF = pvi_vector_field(mdp, many_cores, 0.1)/0.1 - pvi_vector_field(mdp, many_cores, 1e-8)/1e-8
    plt_field(vs, dF)

    plt.savefig('figs/lr_field_diffs.png', dpi=300)
    plt.close()


################################################
#
# # @jit
# def pg_vector_field(mdp, pis, lr):
#     V = lambda logit: value_functional(mdp.P, mdp.r, softmax(logit, axis=1), mdp.discount)
#     logits = [np.log(pi+1e-8) for pi in pis]
#     pg_update_fn = policy_gradient_iteration_logits(mdp, lr)
#     return np.hstack([V(pg_update_fn(logit)) - V(logit) for logit in logits]).T
#
#
# def generate_pg_vs_vi(mdp):
#     print('\nRunning PG vs VI')
#     lr = 0.1
#
#     pis = gen_grid_policies(N=31)
#     vs = polytope(mdp.P, mdp.r, mdp.discount, pis)
#
#     plt.figure(figsize=(16,16))
#
#     # pg
#     dpgs = pg_vector_field(mdp, pis, lr)
#     mags = np.linalg.norm(dpgs, axis=1, keepdims=True)
#     normed_dpg = dpgs/mags
#
#     plt.subplot(2,1,1)
#     plt.quiver(vs[:, 0], vs[:, 1], normed_dpg[:, 0], normed_dpg[:,1], mags)
#     plt.colorbar()
#
#     # vi
#     dvis = vi_vector_field(mdp, pis, lr)
#     mags = np.linalg.norm(dvis, axis=1, keepdims=True)
#     normed_dvis = dvis/mags
#
#     plt.subplot(2,1,2)
#     plt.quiver(vs[:, 0], vs[:, 1], normed_dvis[:, 0], normed_dvis[:,1], mags)
#
#
#     plt.show()


if __name__ == '__main__':
    n_states, n_actions = 2, 2
    mdp = build_random_mdp(n_states, n_actions, 0.5)
    # generate_pvi_vs_vi(mdp)
    # cts_lr_fields(mdp)
    # lr_field_diffs(mdp)
