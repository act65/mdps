"""
Construct and explore the properties of MDPs with structure.
"""
import functools

import numpy
import jax.numpy as np
from jax import grad, jit, jacrev, vmap

import numpy.random as rnd

import mdp.utils as utils
import mdp.search_spaces as ss

def generate_latent_mdp(n_states, n_actions, n_hidden):
    """
    A MDP with a latent space.
    """
    # TODO want to design a structured state space
    # where is it possible to guess state partitions

    P = []
    for _ in range(n_actions):
        # P[s', s, a=a] = UV.T
        U = rnd.random((n_states, n_hidden))
        U = U/np.sum(U, axis=0)
        VT = rnd.random((n_hidden, n_states))
        VT = VT/np.sum(VT, axis=0)
        P.append(np.dot(U, VT))
    P = np.stack(P, axis=-1)

    # NOTE the reward fn should also have this same structure!? how?
    r = rnd.standard_normal((n_states, n_actions))
    d0 = rnd.random((n_states, 1))

    return utils.MDP(n_states, n_actions, P, r, 0.5, d0)

def find_symmetric_mdp(n_states, n_actions, discount, lr=1e-2):
    """
    Approximately find a mdp with ??? symmetry
    """
    model_init = rnd.standard_normal(n_states * n_states * n_actions + n_states * n_actions)
    pis = utils.get_deterministic_policies(n_states, n_actions)
    # pis = [utils.random_policy(n_states, n_actions) for _ in range(100)]
    pis = np.stack(pis)
    # print(pis.shape)
    V = vmap(lambda P, r, pi: utils.value_functional(P, r, pi, discount), in_axes=(None, None, 0))

    def loss_fn(model_params):
        # policy symmetry
        P, r = ss.parse_model_params(n_states, n_actions, model_params)
        return np.sum(np.square(V(utils.softmax(P), r, pis) - V(utils.softmax(P), r, np.flip(pis, 1))))

    # def loss_fn(model_params):
    #     # value symmetry
    #     P, r = ss.parse_model_params(n_states, n_actions, model_params)
    #     vals = V(utils.softmax(P), r, pis)
    #     n = n_states//2
    #     return np.sum(np.square(vals[:, :n] - vals[:, n:]))


    dldp = grad(loss_fn)
    update_fn = lambda model: model - lr*dldp(model)
    init = (model_init, np.zeros_like(model_init))
    model_params, momentum_var = utils.solve(ss.momentum_bundler(update_fn, 0.9), init)[-1]

    P, r = ss.parse_model_params(n_states, n_actions, model_params)
    d0 = rnd.random((n_states, 1))
    return utils.MDP(n_states, n_actions, P, r, discount, d0)


def sample_using_symmetric_prior(S):
    """
    Args:
        S: a similarity matrix. 1 is similar, 0 is not similar.

    Returns:
        X: a matrix where X[i, j] = 1 if S_ij ~= 1. X \in [0,1]^{nS x nS}.
    """

    autG = automorphisms(S)


    return X


if __name__ == '__main__':
    generate_latent_mdp(4, 2, 3)




    # # BUG ?!? doesnt work with n_states = 4/5?!
    # # could be todo with the way the pis are ordered!?
    # n_states, n_actions, discount = 2, 2, 0.5
    # mdp = find_symmetric_mdp(n_states, n_actions, discount, lr=1e-2)
    # V = vmap(lambda pi: utils.value_functional(mdp.P, mdp.r, pi, mdp.discount))
    # # mdp = utils.build_random_mdp(n_states, n_actions, discount)
    # # pi = utils.random_policy(n_states, n_actions)
    # pis = np.stack(utils.get_deterministic_policies(n_states, n_actions))
    # v = V(pis)
    # v_ = V(np.flip(pis, 1))
    # print(np.isclose(np.linalg.norm(v, np.inf, axis=1), np.linalg.norm(v_, np.inf, axis=1), atol=1e-4))
    # print(v, v_)
