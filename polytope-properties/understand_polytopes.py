"""
Want to generate polytopes.

Varying the number of actions, gamma, ...?
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

from polytope_tools import *

def normalise(x, axis):
    return x/np.sum(x, axis=axis, keepdims=True)

def exp_dist(x, lambda_=3.5):  # why 3.5!?
    return lambda_*np.exp(-lambda_*x)

def uniform_simplex(shape):
    # god damn. it's not as simple as I thought to generate uniform distributions on simplexs
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    # http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    return normalise(exp_dist(rnd.random(shape)),axis=1)

def generate_rnd_polytope_densities():
    """
    Randomly sample from `n_action` simplexes.
    And then use the value functional to solve for V.
    """
    generate_rnd_policy = lambda n_states, n_actions: generate_Mpi(n_states,
                                                                   n_actions,
                                                                   uniform_simplex((n_states, n_actions)))
    n_states = 2


    n = 16
    m = 2
    N = 100000

    n_actions = [i for i in range (2, n//m+2) for _ in range(m)]
    plt.figure(figsize=(16, 16))
    count = 0
    for i in range(n//m):
        for _ in range(m):
            count += 1
            P, r = generate_rnd_problem(n_states, n_actions[i])

            # TODO can we calculate this analytically? p(V) = inv(abs(1/det(dVdx))) p(x)
            Vs = np.hstack([value_functional(P, r, generate_rnd_policy(n_states, n_actions[i]), 0.9) for _ in range(N)])

            plt.subplot(n//m,n//m,count+1)
            fig = plt.scatter(Vs[0, :], Vs[1, :],)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)



    plt.tight_layout()
    plt.savefig('figs/polytope_n_actions.png'.format(i))

if __name__ =='__main__':
    generate_rnd_polytope_densities()
