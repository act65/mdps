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


if __name__ == '__main__':
    # BUG ?!? doesnt work with n_states = 4/5?!
    # could be todo with the way the pis are ordered!?
    n_states, n_actions, discount = 2, 2, 0.5
    mdp = find_symmetric_mdp(n_states, n_actions, discount, lr=1e-2)
    V = vmap(lambda pi: utils.value_functional(mdp.P, mdp.r, pi, mdp.discount))
    # mdp = utils.build_random_mdp(n_states, n_actions, discount)
    # pi = utils.random_policy(n_states, n_actions)
    pis = np.stack(utils.get_deterministic_policies(n_states, n_actions))
    v = V(pis)
    v_ = V(np.flip(pis, 1))
    print(np.isclose(np.linalg.norm(v, np.inf, axis=1), np.linalg.norm(v_, np.inf, axis=1), atol=1e-4))
    print(v, v_)
