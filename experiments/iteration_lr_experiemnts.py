import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mdp.utils as utils
import mdp.search_spaces as ss

from jax import vmap

def policy_gradient(mdp, pis, lr):
    lens, pi_stars = [], []

    for pi in pis:
        pi_traj = utils.solve(ss.policy_gradient_iteration_logits(mdp, lr), np.log(pi+1e-8))
        pi_star = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars



def param_policy_gradient(mdp, pis, lr):
    lens, pi_stars = [], []
    core_init = ss.random_parameterised_matrix(2, 2, 32, 8)

    for pi in pis:
        try:
            core_init = ss.approximate(pi, core_init, activation_fn=utils.softmax)
            pi_traj = utils.solve(ss.parameterised_policy_gradient_iteration(mdp, lr/len(core_init)), core_init)
            pi_star = pi_traj[-1]
            L = len(pi_traj)
        except ValueError:
            pi_star = pis[0]
            L = 10000

        pi_stars.append(pi_star)
        lens.append(L)

    return lens, pi_stars


def generate_iteration_figures(mdp, pis, iteration_fn, name):
    """
    How many steps to converge to the optima from different starting points.
    """
    n = 3
    lrs = np.linspace(1e-8, 1, n**2) # 0.5 - 0.00195...
    plt.figure(figsize=(16, 16))
    value = vmap(lambda pi: utils.value_functional(mdp.P, mdp.r, pi, mdp.discount))
    Vs = value(np.stack(pis, axis=0))

    for i, lr in enumerate(lrs):
        print('\n{}: {}\n'.format(i, lr))
        lens, pi_stars = iteration_fn(mdp, pis, lr)

        plt.subplot(n,n,i+1)
        plt.title('Learning rate: {}'.format(lr))
        fig = plt.scatter(Vs[0, :], Vs[1, :], c=lens, s=5)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('figs/iteration-lrs/{}.png'.format(name))

if __name__ =='__main__':
    rnd.seed(41)
    n_states, n_actions = 2, 2
    mdps = [utils.build_random_mdp(n_states, n_actions, 0.5) for _ in range(5)]
    pis = utils.gen_grid_policies(51)

    for i, mdp in enumerate(mdps):
        print('\nMDP {}\n'.format(i))
        generate_iteration_figures(mdp, pis, param_policy_gradient, str(i))
