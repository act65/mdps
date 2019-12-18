import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mdp.utils as utils
from jax import vmap

def plot():
    n_states = 2
    n_actions = 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    value = vmap(lambda pi: utils.value_functional(mdp.P, mdp.r, pi, mdp.discount))

    pis = np.stack(utils.gen_grid_policies(101),axis=0)
    vs = value(pis)
    plt.scatter(vs[:, 0], vs[:, 1], s=10)

    pis = np.stack(utils.get_deterministic_policies(2, 2), axis=0)
    vs = value(pis)
    plt.scatter(vs[:, 0], vs[:, 1], s=10, c='r')

    plt.xlabel('The value of state 1')
    plt.ylabel('The value of state 2')

    plt.title('The value polytope')

    plt.show()

if __name__ == "__main__":
    plot()
