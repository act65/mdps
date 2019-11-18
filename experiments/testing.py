import jax.numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import copy
import functools

import mdp.utils as utils
import mdp.search_spaces as ss

def generate_cvi():
    print('\nRunning PVI vs VI')
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)


    fn = ss.complex_value_iteration(mdp, 0.01)

    Q = rnd.standard_normal((n_states, 1)) + 1j*rnd.standard_normal((n_states, 1))

    results = utils.solve(fn, Q)
    print(results)

if __name__ == '__main__':
    generate_cvi()
