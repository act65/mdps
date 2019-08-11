import numpy as np
import matplotlib.pyplot as plt

from polytope_tools import *

def generate_discounted_polytopes():
    n_states = 2
    n_actions = 2

    ny = 6
    nx = 10
    M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,31)]
    P, r = generate_rnd_problem(n_states,n_actions)
    count = 0
    plt.figure(figsize=(16,16))

    discounts = np.linspace(1e-2, 1-1e-2,nx)

    Vs0 = np.hstack([value_functional(P, r, M_pi, 0) for M_pi in M_pis])

    for i in range(nx-1):
        print(i)
        Vs_tp1 = np.hstack([value_functional(P, r, M_pi, discounts[i+1]) for M_pi in M_pis])
        Vs = np.hstack([value_functional(P, r, M_pi, discounts[i]) for M_pi in M_pis])

        Ws = np.sum((Vs - Vs_tp1)**2, axis=0)

        count += 1
        ax = plt.subplot(2,nx//2,count)
        plt.title('{:.3f} -- {:.3f}'.format(discounts[i], discounts[i+1]))

        fig = plt.scatter(Vs0[0, :], Vs0[1, :], c=Ws)
        # print(M_pis.shape)

    plt.show()
    #
    #     fig.axes.get_xaxis().set_visible(False)
    #     fig.axes.get_yaxis().set_visible(False)
    #
    # plt.tight_layout()
    # plt.savefig('../pictures/figures/discounts-basis.png')

if __name__ == '__main__':
    generate_discounted_polytopes()
