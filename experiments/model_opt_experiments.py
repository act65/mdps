import jax.numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import numpy

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
    apis = np.stack(apis)

    update_fn = model_iteration(mdp, lr, apis)
    params = utils.solve(update_fn, init)
    params = [parse_model_params(mdp.S, mdp.A, p) for p in params]

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
    n_states = 32
    n_actions = 2
    lr = 0.01
    k = 64

    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    init = rnd.standard_normal((mdp.S * mdp.S * mdp.A + mdp.S * mdp.A))

    pi_star = utils.solve(policy_iteration(mdp), utils.softmax(rnd.standard_normal((mdp.S,mdp.A))))[-1]
    print('pi_star\n', pi_star)

    # adversarial pis
    # apis = utils.get_deterministic_policies(mdp.S, mdp.A)
    apis = np.stack([utils.random_det_policy(mdp.S, mdp.A) for _ in range(k)])

    update_fn = model_iteration(mdp, lr, apis)
    params = utils.solve(update_fn, init)
    p_logits, r = parse_model_params(mdp.S, mdp.A, params[-1])
    error = np.mean((utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount) - utils.value_functional(utils.softmax(p_logits), r, pi_star, mdp.discount))**2)
    print('\n', error)
    new_mdp = utils.MDP(mdp.S, mdp.A, utils.softmax(p_logits), r, mdp.discount, mdp.d0)
    pi_star = utils.solve(policy_iteration(new_mdp), utils.softmax(rnd.standard_normal((mdp.S,mdp.A))))[-1]
    print(pi_star)

    apis = np.stack([utils.random_policy(mdp.S, mdp.A) for _ in range(k)])

    update_fn = model_iteration(mdp, lr, apis)
    params = utils.solve(update_fn, init)
    p_logits, r = parse_model_params(mdp.S, mdp.A, params[-1])
    error = np.mean((utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount) - utils.value_functional(utils.softmax(p_logits), r, pi_star, mdp.discount))**2)
    print('\n', error)
    new_mdp = utils.MDP(mdp.S, mdp.A, utils.softmax(p_logits), r, mdp.discount, mdp.d0)
    pi_star = utils.solve(policy_iteration(new_mdp), utils.softmax(rnd.standard_normal((mdp.S,mdp.A))))[-1]
    print(pi_star)


def estimation_err():
    """
    Compare using all deterministic policies versus fewer mixed policies.
    Starts to get interesting in higher dims?



    """
    n_states = 4
    n_actions = 2
    lr = 0.01
    discount = 0.5

    dpis = utils.get_deterministic_policies(n_states, n_actions)
    params = rnd.standard_normal((n_states * n_states * n_actions + n_states * n_actions))

    def value(P, r, pis):
        return np.array([utils.value_functional(P, r, pi, discount) for pi in pis])  # jax doesnt seem to like me changing the batch size to a vmap?!?

    def loss_fn(params, pis):
        p_logits, r = parse_model_params(n_states, n_actions, params)
        return np.sum(value(utils.softmax(p_logits), r, pis)**2)

    dVdp = jit(lambda *x: np.array(grad(loss_fn, 0)(*x))) #,axis=0)
    det_dVdp = dVdp(params, dpis)

    k_estim_err = []
    for k in range(n_states, n_actions**n_states+1, n_states//2):
        print('\n{} det policies. Testing with {}\n'.format(n_actions**n_states, k))
        diffs = []
        for _ in range(6):
            rnd_pis = np.stack([utils.random_det_policy(n_states, n_actions) for _ in range(k)])
            diffs.append(np.max(np.abs(det_dVdp - dVdp(params, rnd_pis))))
        k_estim_err.append(numpy.mean(diffs))


    plt.plot(range(n_states, n_actions**n_states+1, n_states//2), k_estim_err)
    plt.xlabel('Number of randomly sampled policies')
    plt.ylabel('Max error in gradient estimation')
    plt.show()


if __name__ == '__main__':
    # generate_model_iteration()
    # generate_model_cs()
    estimation_err()
