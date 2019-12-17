import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mdp.utils as utils
import mdp.density as density
import mdp.search_spaces as ss

def variance(P, r, pi, discount):
    # var = int_s' int_a p(s'|s, a) pi(a|s)   (r(s,a) + \gamma V(s') + V(s))
    V = utils.value_functional(P, r, pi, discount)[:, 0]
    d = (r[None, :, :] + discount * V[:, None, None] - V[None, :, None])**2
    expected_d = np.einsum('ijk,ijk->j', P * pi[None, :, :], d)  # aka variance
    return np.sum(expected_d)

def loss_fn(pi):  # use grad from policy gradients!?
    return loss

def grad_mag(P, r, pi, discount):
    P_pi = np.einsum('ijk,jk->ij', P, pi) #np.dot(M_pi, P)
    r_pi = np.einsum('jk,jk->j', pi, r)  #np.dot(M_pi, r)

    J = density.value_jacobian(r_pi, P_pi, discount)
    return np.linalg.norm(J)


def generate_snr_map():
    n_states, n_actions = 2, 3
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    # pis = utils.gen_grid_policies(11)
    pis = [utils.random_policy(n_states, n_actions) for _ in range(512)]
    Vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)

    mags = [grad_mag(mdp.P, mdp.r, pi, mdp.discount) for pi in pis]
    uncert = [variance(mdp.P, mdp.r, pi, mdp.discount) for pi in pis]

    snr = [s/n for s,n in zip(mags, uncert)]

    plt.subplot(3, 1, 1)
    plt.title('Magnitude')
    plt.scatter(Vs[:, 0], Vs[:, 1], c=mags)

    plt.subplot(3, 1, 2)
    plt.title('Variance')
    plt.scatter(Vs[:, 0], Vs[:, 1], c=uncert)

    plt.subplot(3, 1, 3)
    plt.title('SNR')
    plt.scatter(Vs[:, 0], Vs[:, 1], c=snr)
    plt.show()

if __name__ == "__main__":
    generate_snr_map()
