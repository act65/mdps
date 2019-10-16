import jax.numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import copy
import functools

import mdp
import mdp.utils
from mdp.search_spaces import *



def generate_model_iteration():
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = utils.gen_grid_policies(7)

    init = rnd.standard_normal((mdp.S * mdp.S * mdp.A + mdp.S * mdp.A))  # needs its own init. alternatively could find init that matches value of other inits?!?

    vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)


    plt.figure(figsize=(16,16))
    plt.scatter(vs[:, 0], vs[:, 1], c='b', s=10, alpha=0.75)

    lr = 0.01
    pi_star = utils.solve(policy_iteration(mdp), utils.softmax(rnd.standard_normal((mdp.S,mdp.A))))[-1]

    # adversarial pis
    apis = utils.get_deterministic_policies(mdp.S, mdp.A)

    update_fn = model_iteration(mdp, lr, apis)
    params = utils.solve(update_fn, init)
    params = [parse_model_params(mdp, p) for p in params]

    vs = np.vstack([utils.value_functional(utils.softmax(p_logits), r, pi_star, mdp.discount).T for p_logits, r in params])

    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='PG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    p_logits, r = params[-1]
    vs = utils.polytope(utils.softmax(p_logits), r, mdp.discount, pis)
    plt.scatter(vs[:, 0], vs[:, 1], c='r', s=10, alpha=0.75)
    plt.title('Model iteration')
    plt.xlabel('Value of state 1')
    plt.ylabel('Value of state 2')

    # plt.show()
    plt.savefig('figs/model_iteration_1.png')

    learned_mdp = utils.MDP(mdp.S, mdp.A, utils.softmax(p_logits), r, mdp.discount, mdp.d0)
    pi_star_approx = utils.solve(policy_iteration(learned_mdp), utils.softmax(rnd.standard_normal((mdp.S,mdp.A))))[-1]
    print(pi_star_approx, '\n', pi_star)

def generate_model_cs():
    """
    Compare using all deterministic policies versus fewer mixed policies.
    Starts to get interesting in higher dims?



    """
    n_actions = 2
    for n_states in range(2, 12):

        # adversarial pis
        # pis = utils.get_deterministic_policies(mdp.S, mdp.A)
        apis = utils.get_random_policies(mdp.S, mdp.A, N)


        update_fn = model_iteration(mdp, lr, apis)
        params = utils.solve(update_fn, init)
        p_logits, r = parse_model_params(mdp, params[-1])
        error = (utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount) - utils.value_functional(utils.softmax(p_logits), r, pi_star, mdp.discount))**2
        # which pis should we evaluate under? many?


if __name__ == '__main__':
    generate_model_iteration()
    # generate_model_cs()
