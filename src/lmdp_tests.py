import numpy as np

from lmdps import *

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

        pi = softmax(r, axis=1)  # exp Q vals w gamma = 0
        # a distribution over actions
        assert np.isclose(np.sum(pi, axis=1), np.ones((2,))).all()

        p, q = mdp_encoder(P, r)

        print('q', q)
        P_pi = np.einsum('ijk,jk->ij', P, pi)

        print('p', p)
        print('P_pi', P_pi)

        # the unconstrained dynamics with deterministic transitions,
        # are the same was using a gamma = 0 boltzman Q vals
        print("exp(r) is close to p? {}".format(np.isclose(p, P_pi, atol=1e-4).all()))

        # r(s, a) = q(s) - KL(P(. | s, a) || p(. | s))
        ce = numpy.zeros((2, 2))
        for j in range(2):
            for k in range(2): # actions
                ce[j, k] = CE(P[:, j, k], p[:, j])

        r_approx = q[:, np.newaxis] - ce

        print(np.around(r, 3))
        print(np.around(r_approx, 3))
        print('r ~= q - CE(P || p): {}'.format(np.isclose(r, r_approx, atol=1e-3).all()))
        print('\n\n')

    @staticmethod
    def random_test():
        """
        Explore how the unconstrained dynamics in a random setting.
        """
        n_states, n_actions = 12, 3
        P, r = rnd_mdp(n_states, n_actions)

        # a distribution over future states
        assert np.isclose(np.sum(P, axis=0), np.ones((n_states, n_actions))).all()

        p, q = mdp_encoder(P, r)

        # print('q', q)
        # print('p', p)

        # r(s, a) = q(s) - KL(P(. | s, a) || p(. | s))
        # TODO how to do with matrices!?
        # kl = - (np.einsum('ijk,ij->jk', P, np.log(p)) - np.einsum('ijk,ijk->jk', P, np.log(P)))
        ce = numpy.zeros((n_states, n_actions))
        for j in range(n_states):
            for k in range(n_actions): # actions
                ce[j, k] = CE(P[:, j, k], p[:, j])

        r_approx = q[:, np.newaxis] - ce

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

if __name__ == "__main__":
    # TestMDPEmbeddeding()
    # TestLMDPSolver()
    DecodeLMDPControl()
