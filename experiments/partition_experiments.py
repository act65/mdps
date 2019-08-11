import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import src.utils as utils
import src.search_spaces as ss

# TODO could also do something similar with value iteration or PG?!?
# what about PI under noise?

def clip_solver_traj(traj):
    if np.isclose(traj[-1], traj[-2], 1e-8).all():
        return traj[:-1]
    else:
        return traj

def policy_iteration_partitions(mdp, pis):
    """
    For each policy on a uniform grid.
    Use that policy as an init and solve the MDP.
    Lens = how many steps are required to solve the MDP.
    """
    lens, pi_stars = [], []
    init = utils.softmax(rnd.standard_normal((mdp.S, mdp.A)), axis=1)

    for pi in pis:
        pi_traj = clip_solver_traj(utils.solve(ss.policy_iteration(mdp), init))
        pi_star = pi_traj[-1]

        pi_stars.append(pi_star)
        lens.append(len(pi_traj))

    return lens, pi_stars

def generate_partition_figures():
    """
    Print how many gpi steps are required to converge to the optima wrt pi/v.
    """
    n_states, n_actions = 2, 2
    pis = utils.gen_grid_policies(21)

    nx = 3
    ny = 3

    plt.figure(figsize=(16, 16))
    for i in range(nx*ny):
        print('\n', i)
        mdp = utils.build_random_mdp(n_states, n_actions, 0.9)
        lens, pi_stars = policy_iteration_partitions(mdp, pis)
        Vs = np.hstack([utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)
                        for pi in pis])

        plt.subplot(nx,ny,i+1)
        fig = plt.scatter(Vs[0, :], Vs[1, :], c=lens, s=1)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    # plt.savefig('../pictures/figures/gpi-partitions.png')
    plt.show()

if __name__ =='__main__':
    generate_partition_figures()
