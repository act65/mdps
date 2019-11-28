import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mdp.utils as utils
import mdp.search_spaces as ss

def clip_solver_traj(traj):
    if np.isclose(traj[-1], traj[-2], 1e-8).all():
        return traj[:-1]
    else:
        return traj

"""
For each policy on a uniform grid.
Use that policy as an init and solve the MDP.
Lens = how many steps are required to solve the MDP.
"""

def policy_iteration(mdp, pis):
    # pi_star = utils.solve(ss.policy_iteration(mdp), pis[0])[-1]

    lens, pi_stars = [], []

    for pi in pis:
        pi_traj = clip_solver_traj(utils.solve(ss.policy_iteration(mdp), pi))
        pi_star = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def policy_gradient(mdp, pis):
    lens, pi_stars = [], []

    for pi in pis:
        pi_traj = utils.solve(ss.policy_gradient_iteration_logits(mdp, 0.01), np.log(pi+1e-8))
        pi_star = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def mom_policy_gradient(mdp, pis):
    lens, pi_stars = [], []

    for pi in pis:
        pi_traj = utils.solve(ss.momentum_bundler(ss.policy_gradient_iteration_logits(mdp, 0.01), 0.9), (np.log(pi+1e-8), np.zeros_like(pi)))
        pi_star, _ = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def param_policy_gradient(mdp, pis):
    lens, pi_stars = [], []
    core_init = ss.random_parameterised_matrix(2, 2, 32, 8)

    for pi in pis:
        core_init = ss.approximate(pi, core_init, activation_fn=utils.softmax)
        pi_traj = utils.solve(ss.parameterised_policy_gradient_iteration(mdp, 0.01/len(core_init)), core_init)
        pi_star = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def param_mom_policy_gradient(mdp, pis):
    lens, pi_stars = [], []
    core_init = ss.random_parameterised_matrix(2, 2, 32, 8)

    for pi in pis:
        core_init = ss.approximate(pi, core_init, activation_fn=utils.softmax)
        init = (core_init, [np.zeros_like(c) for c in core_init])
        pi_traj = utils.solve(ss.momentum_bundler(ss.parameterised_policy_gradient_iteration(mdp, 0.01/len(core_init)), 0.9), init)
        pi_star, _ = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def value_iteration(mdp, pis):
    lens, pi_stars = [], []

    for pi in pis:
        init_V = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)
        pi_traj = utils.solve(ss.value_iteration(mdp, 0.01), init_V)
        pi_star = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def mom_value_iteration(mdp, pis):
    lens, pi_stars = [], []


    for pi in pis:
        init_V = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)
        pi_traj = utils.solve(ss.momentum_bundler(ss.value_iteration(mdp, 0.01), 0.9), (init_V, np.zeros_like(init_V)))
        pi_star, _ = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def param_value_iteration(mdp, pis):
    # hypothesis. we are going to see some weirdness in the mom partitions.
    # oscillations will depend on shape of the polytope?!?
    lens, pi_stars = [], []

    core_init = ss.random_parameterised_matrix(2, 2, 32, 4)

    for pi in pis:
        init_V = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)
        core_init = ss.approximate(init_V, core_init)
        params = utils.solve(ss.parameterised_value_iteration(mdp, 0.01/len(core_init)), core_init)
        pi_star = params[-1]

        pi_stars.append(pi_star)
        lens.append(len(params))

    return lens, pi_stars

def mom_param_value_iteration(mdp, pis):
    lens, pi_stars = [], []

    core_init = ss.random_parameterised_matrix(2, 2, 32, 4)

    for pi in pis:
        init_V = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)
        core_init = ss.approximate(init_V, core_init)
        params = utils.solve(ss.momentum_bundler(ss.parameterised_value_iteration(mdp, 0.01/len(core_init)), 0.8), (core_init, [np.zeros_like(c) for c in core_init]))
        pi_star, _ = params[-1]

        pi_stars.append(pi_star)
        lens.append(len(params))

    return lens, pi_stars

def generate_iteration_figures(mdps, pis, iteration_fn, name):
    """
    How many steps to converge to the optima from different starting points.
    """
    n = np.sqrt(len(mdps))
    plt.figure(figsize=(16, 16))
    for i, mdp in enumerate(mdps):
        print(i)
        Vs = np.hstack([utils.value_functional(mdp.P, mdp.r, pi, mdp.discount) for pi in pis])
        lens, pi_stars = iteration_fn(mdp, pis)

        plt.subplot(n,n,i+1)
        fig = plt.scatter(Vs[0, :], Vs[1, :], c=lens, s=5)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/iterations/{}.png'.format(name))

if __name__ =='__main__':
    rnd.seed(41)
    n_states, n_actions = 2, 2
    mdps = [utils.build_random_mdp(n_states, n_actions, 0.5) for _ in range(3*3)]
    pis = utils.gen_grid_policies(31)

    iteration_fns = [
        # value_iteration,
        # param_value_iteration,
        # mom_value_iteration,
        # mom_param_value_iteration,
        # policy_iteration_partitions,
        # policy_gradient,
        param_policy_gradient,
        mom_policy_gradient,
        param_mom_policy_gradient,
    ]

    for fn in iteration_fns:
        print(fn.__name__)
        generate_iteration_figures(mdps, pis, fn, fn.__name__)
