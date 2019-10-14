import jax.numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import copy
import functools

import mdp
import mdp.utils
from mdp.search_spaces import *



def generate_model_iteration(mdp, init):
    pi_star = utils.solve(policy_iteration(mdp), utils.softmax(rnd.standard_normal((mdp.S,mdp.A))))[-1]
    update_fn = model_iteration(mdp, 0.01)
    logits = utils.solve(update_fn, init)

    vs = np.vstack([utils.value_functional(utils.softmax(logit), mdp.r, pi_star, mdp.discount).T for logit in logits])

    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='PG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    return utils.softmax(logits[-1])



if __name__ == '__main__':
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = utils.gen_grid_policies(9)

    init = rnd.standard_normal((mdp.S, mdp.S, mdp.A))  # needs its own init. alternatively could find init that matches value of other inits?!?

    vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)



    plt.figure(figsize=(16,16))
    plt.scatter(vs[:, 0], vs[:, 1], c='b', s=10, alpha=0.75)
    model = generate_model_iteration(mdp, init)
    vs = utils.polytope(model, mdp.r, mdp.discount, pis)
    plt.scatter(vs[:, 0], vs[:, 1], c='r', s=10, alpha=0.75)

    plt.show()
