"""
Want to generate pics of value under approximate transition fns

Question is. How likely is this error to push GPI to an erroneous solution?
Also. How would this scale in many dimension!?
More likely to be a partition that is 'close'.
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

from polytope_tools import *

def ce_losses(xs, y):
    return 1


def generate_noisy_transitions():
    n_states = 2
    n_actions = 2
    discount = 0.75
    P, r = generate_rnd_problem(n_states, n_actions)

    M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,31)]
    Vs = np.hstack([value_functional(P, r, M_pi, discount) for M_pi in M_pis])

    m = 3
    stddev = np.logspace(-4, -1, m*m)
    # print(stddev)
    # raise SystemExit
    for i in range(m*m):
        plt.subplot(m,m,i+1)
        fig = plt.scatter(Vs[0, :], Vs[1, :], alpha=1, s=1)

        plt.title('Noise mag: {:.3f}'.format(stddev[i]))
        n = 2000
        idx = int(6.5*32)
        M_pi = M_pis[idx]
        Ps_hat = [P + stddev[i]*(2*np.random.random(P.shape)-1) for _ in range(n)]

        # TODO want an intuition for how the accuracy of the learned P
        # matches to the expected amount of error in the estimation
        # plt.title('Noise mag: {:.3f}'.format(ce_losses(P, Ps_hat)))

        Vs_hat = np.hstack([value_functional(Pi, r, M_pi, discount) for Pi in Ps_hat])
        fig = plt.scatter(Vs_hat[0, :], Vs_hat[1, :], alpha=0.95, s=1, c='g')

        plt.scatter(Vs[0, idx], Vs[1, idx], alpha=0.75, s=30, c='r')

        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig('../pictures/figures/noisy-transitions.png')


if __name__ =='__main__':
    generate_noisy_transitions()
