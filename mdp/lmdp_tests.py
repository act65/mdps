import mdp.utils as utils
from mdp.lmdps import *

import numpy as np

class TestMDPEmbeddeding():
    def __init__(self):
        self.simple_test()
        self.random_test()

    @staticmethod
    def simple_test():
        """
        Explore how the unconstrained dynamics in a simple setting.
        """
        # What about when p(s'| s) = 0, is not possible under the true dynamics?!
        r = np.array([
            [1, 0],
            [0, 0]
        ])

        # Indexed by [s' x s x a]
        # ensure we have a distribution over s'
        p000 = 1
        p100 = 1 - p000

        p001 = 0
        p101 = 1 - p001

        p010 = 0
        p110 = 1 - p010

        p011 = 1
        p111 = 1 - p011

        P = np.array([
            [[p000, p001],
             [p010, p011]],
            [[p100, p101],
             [p110, p111]],
        ])
        # BUG ??? only seems to work for deterministic transitions!?
        # oh, this is because deterministic transitions satisfy the row rank requirement??!
        # P = np.random.random((2, 2, 2))
        # P = P/np.sum(P, axis=0)

        # a distribution over future states
        assert np.isclose(np.sum(P, axis=0), np.ones((2,2))).all()

        pi = utils.softmax(r, axis=1)  # exp Q vals w gamma = 0
        # a distribution over actions
        assert np.isclose(np.sum(pi, axis=1), np.ones((2,))).all()

        p, q = mdp_encoder(P, r)

        print('q', q)
        print('p', p)
        print('P', P)
        P_pi = np.einsum('ijk,jk->ij', P, pi)
        print('P_pi', P_pi)

        # the unconstrained dynamics with deterministic transitions,
        # are the same was using a gamma = 0 boltzman Q vals
        print("exp(r) is close to p? {}".format(np.isclose(p, P_pi, atol=1e-4).all()))

        # r(s, a) = q(s) - KL(P(. | s, a) || p(. | s))
        ce = numpy.zeros((2, 2))
        for j in range(2):
            for k in range(2): # actions
                ce[j, k] = CE(P[:, j, k], p[:, j])

        r_approx = q[:, np.newaxis] + ce

        print(np.around(r, 3))
        print(np.around(r_approx, 3))
        print('r ~= q - CE(P || p): {}'.format(np.isclose(r, r_approx, atol=1e-2).all()))
        print('\n\n')

    @staticmethod
    def random_test():
        """
        Explore how the unconstrained dynamics in a random setting.
        """
        n_states, n_actions = 3, 2
        mdp = utils.build_random_mdp(n_states, n_actions, 0.9)
        P = mdp.P
        r = mdp.r

        # a distribution over future states
        assert np.isclose(np.sum(P, axis=0), np.ones((n_states, n_actions))).all()

        p, q = mdp_encoder(P, r)

        # print('P', P)
        # print('r', r)
        # print('q', q)
        # print('p', p)

        # r(s, a) = q(s) - KL(P(. | s, a) || p(. | s))
        # TODO how to do with matrices!?
        # kl = - (np.einsum('ijk,ij->jk', P, np.log(p)) - np.einsum('ijk,ijk->jk', P, np.log(P)))
        ce = numpy.zeros((n_states, n_actions))
        for j in range(n_states):
            for k in range(n_actions): # actions
                ce[j, k] = CE(P[:, j, k], p[:, j])

        r_approx = q[:, np.newaxis] + ce

        print('r', np.around(r, 3), r.shape)
        print('r_approx', np.around(r_approx, 3), r_approx.shape)
        print('r ~= q - CE(P || p): {}'.format(np.isclose(r, r_approx, atol=1e-3).all()))

class TestLMDPSolver():
    def __init__(self):
        self.simple_solve_test()
        self.random_solve_test()

    @staticmethod
    def simple_solve_test():
        """
        Simple test. Does it pick the best state?
        """
        p = np.array([
            [0.75, 0.5],
            [0.25, 0.5]
        ])
        q = np.array([1, 0])
        u, v = lmdp_solver(p, q, 0.9)
        assert np.argmax(v) == 0

    @staticmethod
    def random_solve_test():
        """
        Want to set up a env that will test long term value over short term rewards.
        """
        n_states, n_actions = 12, 3
        p, q = rnd_lmdp(n_states, n_actions)
        u, v = lmdp_solver(p, q, 0.99)
        print(u)
        print(v)

    def long_term_test():
        pass

class DecodeLMDPControl():
    def __init__(self):
        # self.test_decoder_simple()
        # self.test_decoder_rnd()
        self.option_decoder()

    @staticmethod
    def test_decoder_simple():
        # Indexed by [s' x s x a]
        # ensure we have a distribution over s'
        p000 = 1
        p100 = 1 - p000

        p001 = 0
        p101 = 1 - p001

        p010 = 0
        p110 = 1 - p010

        p011 = 1
        p111 = 1 - p011

        P = np.array([
            [[p000, p001],
             [p010, p011]],
            [[p100, p101],
             [p110, p111]],
        ])

        u = np.array([
            [0.95, 0.25],
            [0.05, 0.75]
        ])

        pi = lmdp_decoder(u, P, lr=1)
        P_pi = np.einsum('ijk,jk->ij', P, pi)

        assert np.isclose(P_pi, u, atol=1e-4).all()
        print(P_pi)
        print(u)

    @staticmethod
    def test_decoder_rnd():
        n_states = 6
        n_actions = 6

        P = rnd.random((n_states, n_states, n_actions))
        P /= P.sum(0, keepdims=True)

        u = rnd.random((n_states, n_states))
        u /= u.sum(0, keepdims=True)

        pi = lmdp_decoder(u, P, lr=1)
        P_pi = np.einsum('ijk,jk->ij', P, pi)

        print(P_pi)
        print(u)
        print(KL(P_pi,u))
        assert np.isclose(P_pi, u, atol=1e-2).all()

    @staticmethod
    def option_decoder():
        n_states = 32
        n_actions = 4

        P = rnd.random((n_states, n_states, n_actions))
        P /= P.sum(0, keepdims=True)

        u = rnd.random((n_states, n_states))
        u /= u.sum(0, keepdims=True)

        pi = lmdp_option_decoder(u, P)
        print(pi)

def construct_chain(n_states, r_max):
    n_actions = 2
    r = 0*np.ones((n_states, n_actions))
    # r[1, :] = r_max//n_states
    r[n_states-3, 1] = -r_max/2
    r[n_states-2, 1] = r_max

    r[0, :] = 0
    r[n_states-1, :] = 0

    p = 0.9
    P = np.zeros((n_states, n_states, n_actions))

    # absorbing states
    P[0, 0, 0] = 1
    P[1, 0, 0] = 0
    P[0, 0, 1] = 1
    P[1, 0, 1] = 0

    m = n_states-1
    P[m, m, 0] = 1
    P[m-1, m, 0] = 0
    P[m, m, 1] = 1
    P[m-1, m, 1] = 0

    # go left
    for i in range(1, n_states-1):
        P[i-1, i, 0] = p
        P[i+1, i, 0] = 1-p
    # go right
    for i in range(1, n_states-1):
        P[i+1, i, 1] = p
        P[i-1, i, 1] = 1-p

    for a in range(n_actions):
        assert np.isclose(np.sum(P[:, :, a], axis=0), np.ones((n_states, ))).all()

    return P, r

class DiscountingTest():
    def __init__(self):
        self.chain_test()

    @staticmethod
    def chain_test():
        n_states = 16
        P, r = construct_chain(n_states, 20)

        p, q = mdp_encoder(P, r)
        u, v = lmdp_solver(p, q, 0.75)

        plt.figure(figsize=(16,16))

        plt.subplot(2, 2, 1)
        plt.title('P: transition function')
        plt.imshow(np.sum(P, axis=-1))

        plt.subplot(2, 2, 2)
        plt.title('r: reward function')
        plt.imshow(r)

        plt.subplot(2,2,3)
        plt.title('p: unconstrained dynamics')
        plt.imshow(p)

        plt.subplot(2, 2, 4)
        plt.title('u: optimal control')
        plt.imshow(u)

        plt.show()

        """
        PROBLEM!
        hmm. maybe this would be solved with finite horizon MDPs?!
        """

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # TestMDPEmbeddeding()
    # TestLMDPSolver()
    # DecodeLMDPControl()
    DiscountingTest()
