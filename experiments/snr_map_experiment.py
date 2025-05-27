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
    plt.savefig('figs/snr_map.png')
    plt.close()


def est_var_R(mdp, pi, n=60, T=25):
    Rs = []
    for _ in range(n):
        s, a, r = tuple(zip(*utils.rollout(mdp.P, mdp.r, mdp.d0, pi, T)))
        R = utils.discounted_rewards(r, mdp.discount)
        Rs.append(R)

    return np.var(Rs)

def emp_est_snr_graph():
    n_states, n_actions = 12, 3
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = [utils.random_policy(n_states, n_actions) for _ in range(100)]

    vs = []
    hs = []
    for i, pi in enumerate(pis):
        print('\r{}'.format(i), end='',flush=True)

        # try:
        vs.append(est_var_R(mdp, pi))
        hs.append(utils.entropy(pi))
        # except ValueError as err:
        #     print(err)

    plt.scatter(hs, vs)
    plt.savefig('figs/empirical_estimated_snr_graph.png')
    plt.close()

def emp_est_snr_map():
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = utils.gen_grid_policies(5)
    vals = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)

    vars = []
    hs = []
    for i, pi in enumerate(pis):
        print('\r{}'.format(i), end='',flush=True)

        vars.append(est_var_R(mdp, pi))
        hs.append(utils.entropy(pi))

    plt.subplot(2,1,1)
    plt.scatter(vals[:, 0], vals[:, 1], c=hs)
    plt.subplot(2,1,2)
    plt.scatter(vals[:, 0], vals[:, 1], c=vars)
    # plt.subplot(3,1,1)
    # plt.scatter(vals[:, 0], vals[:, 0], c=hs)
    plt.savefig('figs/empirical_estimated_snr_map.png')
    plt.close()


if __name__ == "__main__":
    # generate_snr_map()
    # emp_est_snr_map()
    emp_est_snr_graph()
