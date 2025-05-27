"""
Want to be able to visualise how different abstractions yield exploration.

Idea; use the state visitation probability (or its generalisation).

- but, with what learning algorithm? what type of exploration?
- what about for state-action abstraction?
-

http://proceedings.mlr.press/v97/hazan19a/hazan19a.pdf


Side note. Some policies will explore better than others.
Could plot the entropy of the visitation distributions on the polytope.
"""


import numpy as np

import mdp.utils as utils
import matplotlib.pyplot as plt

def state_action_visitation_distribution(mdp, pi):
    """
    How likely are you to go from one state-action to another state-action?!
    """
    P_s_sa = np.reshape(np.einsum('ijk,jk->ijk', mdp.P, pi), (mdp.S, mdp.S*mdp.A))
    P_sa_sa = np.reshape(np.einsum('ik,ij->ijk', P_s_sa, pi), (mdp.S*mdp.A, mdp.S*mdp.A))
    return np.linalg.inv(np.eye(mdp.S*mdp.A)-mdp.discount * P_sa_sa)

def state_action_vis():
    # want to pick policies that maximise exploration.
    # but. how to solve for this analytically?! not sure this is going to work...
    # unless? is there a way to analytically set pi = 1/visitation?!
    # if we iterate. estimate visitation under pi, set pi = 1/visitaiton.
    # does it converge? where does it converge?
    # it shouldnt converge?!?


    mdp = utils.build_random_mdp(12, 2, 0.5)
    pi = utils.random_policy(mdp.S, mdp.A)
    v_sa_sa = state_action_visitation_distribution(mdp, pi)

    # sum over initial conditions to get discounted state-action visitation probability
    d0_sa = np.reshape(np.einsum('jk,jl->jk', pi, mdp.d0), (mdp.S*mdp.A,))
    ps = np.einsum('ik,k->i', v_sa_sa, d0_sa)

    plt.imshow(v_sa_sa)
    plt.show()

if __name__ == '__main__':
    state_action_vis()
