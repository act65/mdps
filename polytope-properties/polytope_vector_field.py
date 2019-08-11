"""
Want to generate vector fields

NOTE. Want a measure of how easy a space is to optimise.
The homogeneity of the gradients. !!!
For each neighborhood, take the laplacian!? / the det!?
"""
import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

from polytope_tools import *

def generate_field():
    """
    """
    n_states = 2
    n_actions = 2
    discount = 0.8

    n = 3
    for i in range(n*n):
        plt.subplot(n, n, i+1)
        P, r = generate_rnd_problem(n_states, n_actions)
        M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,21)]
        Vs = np.squeeze(np.stack([value_functional(P, r, M_pi, discount) for M_pi in M_pis], axis=0))
        dVs = np.sum(np.stack([value_jacobian(np.dot(M_pi, r), np.dot(M_pi, P), discount) for M_pi in M_pis], axis=0), axis=1)  # QUESTION 1 or 2?
        n_dVs = np.stack([v/np.linalg.norm(v) for v in dVs], axis=0)

        fig = plt.quiver(Vs[:, 0], Vs[:, 1], n_dVs[:, 0], n_dVs[:, 1], np.sum(dVs,axis=1))
        #
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.colorbar()

    plt.tight_layout()
    # plt.savefig('../pictures/figures/polytope_vector_fields.png')
    plt.show()


def distribution_of_grads():
    """
    """
    n_states = 2
    n_actions = 2
    discount = 0.8

    n = 10000
    vals = []
    M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,4)]
    for i in range(n):
        P, r = generate_rnd_problem(n_states, n_actions)
        dVs = np.sum(np.stack([value_jacobian(np.dot(M_pi, r), np.dot(M_pi, P), discount) for M_pi in M_pis]), axis=1)  # 1 or 2?
        dVs = list(sorted(dVs, key=np.linalg.norm))
        vals.append(np.linalg.norm(dVs[0] - dVs[-1]))

    # print(vals)
    plt.hist(vals, bins=100)
    plt.show()


def soft_functional(P, r, M_pi, discount, alpha):
    V = value_functional(P, r, M_pi, discount)
    pi = get_pi(M_pi)
    return V - alpha * entropy(pi)

def soft_jacobian(M_pi, P, r, discount, alpha):
    """
    Calculate the jacobian of a entropy regularised MDP.
    L(pi) = V(pi) - H(pi)
    """
    dVdpi = value_jacobian(np.dot(M_pi, r), np.dot(M_pi, P), discount)
    dHdpi = entropy_jacobian(get_pi(M_pi)+1e-8)
    # return dVdpi - alpha*dHdpi
    return - alpha*dHdpi

def entropy(p):
    return -np.sum(p * np.log(p+1e-8))

def generate_entropy_fields():
    """
    ??? getting weird pics. not sure what is happening here.
    """
    n_states = 2
    n_actions = 2
    discount = 0.7

    P, r = generate_rnd_problem(n_states, n_actions)
    M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,21)]
    n = 3

    x = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(x,x)
    Z = np.stack([X.ravel(), Y.ravel()], axis=1)

    alphas = np.linspace(0.1, 1, n*n)
    for i in range(n*n):
        plt.subplot(n, n, i+1)


        Vs = np.squeeze(np.stack([value_functional(P, r, M_pi, discount) for M_pi in M_pis], axis=0))
        dVs = np.sum(np.stack([soft_jacobian(M_pi, P, r, discount, alphas[i]) for M_pi in M_pis], axis=0), axis=-1)  # QUESTION 1 or 2?
        n_dVs = np.stack([v/np.linalg.norm(v) for v in dVs], axis=0)

        fig = plt.quiver(Z[:, 0], Z[:, 1], n_dVs[:, 0], n_dVs[:, 1], np.sum(dVs,axis=1))
        # BUG. soft jacobian is wrong!?

        # fig = plt.quiver(Vs[:, 0], Vs[:, 1], n_dVs[:, 0], n_dVs[:, 1], np.sum(dVs,axis=1))
        plt.title('Alpha={:.3f}'.format(alphas[i]))


    plt.tight_layout()
    # plt.savefig('../pictures/figures/polytope_entropy_fields.png')
    plt.show()

if __name__ =='__main__':
    # generate_field()
    # distribution_of_grads()
    generate_entropy_fields()
