"""
Want to generate entropy polytope plots.
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

import mdp.utils as utils
from mdp.density import *

def generate_polytope_densities():
    n_states, n_actions = 2, 2
    pis = utils.gen_grid_policies(41)

    nx = 3
    ny = 3
    plt.figure(figsize=(16, 16))

    for i in range(nx*ny):
        print(i)
        mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
        Vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)
        # just set all to be the same probability
        p_pi = 0.1
        pVs = [density_value_functional(p_pi, mdp.P, mdp.r, pi, mdp.discount) for pi in pis]

        plt.subplot(nx,ny,i+1)

        fig = plt.scatter(Vs[:, 0], Vs[:, 1], c=pVs)
        # plt.colorbar()
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    # plt.show() # Replaced with savefig and close
    plt.savefig('figs/polytope_densities.png')
    plt.close()

if __name__ =='__main__':
    generate_polytope_densities()
