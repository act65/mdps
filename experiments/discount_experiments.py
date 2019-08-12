import numpy as np
import matplotlib.pyplot as plt

import src.utils as utils

def hyperbolic_polytope():
        # https://arxiv.org/abs/1902.06865
        n_states, n_actions = 2, 2
        N = 21
        pis = utils.gen_grid_policies(N)
        mdp = utils.build_random_mdp(n_states, n_actions, None)

        n = 10
        discounts = np.linspace(0.1, 1-1e-4, n)
        Vs = []
        for discount in discounts:
            Vs.append((1-discount)*utils.polytope(mdp.P, mdp.r, discount, pis))

        h_V = sum(Vs)/n

        plt.subplot(2, 1, 1)
        plt.scatter(h_V[:, 0], h_V[:, 1])
        plt.subplot(2, 1, 2)
        V = (1-0.9)*utils.polytope(mdp.P, mdp.r, 0.9, pis)
        plt.scatter(V[:, 0], V[:, 1])
        plt.show()

if __name__ == '__main__':
    hyperbolic_polytope()
