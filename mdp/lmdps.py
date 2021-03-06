import numpy
from numpy import linalg
import numpy.random as rnd

import jax.numpy as np
from jax import grad, jit

import mdp.utils as utils

def rnd_lmdp(n_states, n_actions):
    p = rnd.random((n_states, n_states))
    p = p/p.sum(0)
    q = rnd.random((n_states, 1))
    return p, q

def mdp_encoder(P, r):
    """
    Args:
        P (np.array): The transition function. shape = [n_states, n_states, n_actions]
        r (np.array): The reward function. shape = [n_states, n_actions]

    Returns:
        p (np.array): the uncontrolled transition matrix
        q (np.array): the pseudo reward

    Needs to be solved for every state.
    """
    # QUESTION Are there similarities between the action embeddings in each state?
    # QUESTION How does this embedding change the set of policies that can be represented? Does this transformation preserve the optima?
    def embed_state(idx_x):
        """
        For each state, we have a system of |A| linear equations.
        Each eqn requiring that there exists u(.|a), a in A, s.t.;
        - p(s' | s, a) = u(s'|a).p(s'|s)
        - r(s, a) = q(s) - KL(u(.|a)p(.|s), p(.| s))

        This ensures that p, q are able to `represent`#?!?# the original dynamics and
        reward.

        See supplementary material of todorov 2009 for more info
        https://www.pnas.org/content/106/28/11478
        """
        # b_a = r(x, a) ## - E_s'~p(s' | s, a) log( p(s' | s, a) / p(x' | x))
        b = r[idx_x, :] # - np.sum(P[:, idx_x, :] * np.log(P[:, idx_x, :]+1e-8), axis=0) # swap +/- of 2nd term
        # D_ax' = p(x' | x, a)
        D = P[:, idx_x, :]

        # D(q.1 - m) = b,
        c = np.dot(b, linalg.pinv(D))
        # min c
        # q = -np.log(np.sum(np.exp(-c)))
        q = np.log(np.sum(np.exp(c)))  # TODO why?!

        # c = q.1 - m
        # m = q - c
        m = c - q  # TODO why?!

        # p = exp(c + log(sum exp c) ). weird.
        p = np.exp(m)

        # p should be a distribution
        # QUESTION where does p get normalised!?!?
        err = np.isclose(p.sum(0), np.ones(p.shape[0]), 1e-3)
        if not err.any():
            print(err)
            raise ValueError('p is not normalised')

        return p, np.array([q])

    # TODO, can solve these in parallel.
    # maybe even share some knowledge???!
    pnqs = [embed_state(i) for i in range(P.shape[0])]
    p, q = tuple([np.stack(val, axis=1) for val in zip(*pnqs)])
    return p, np.squeeze(q)

def KL(P, Q):
    return -np.sum(P*np.log((Q+1e-8)/(P+1e-8)))

def CE(P, Q):
    return np.sum(P*np.log(Q+1e-8))

@jit
def linear_bellman_operator(p, q, z, discount):
    """
    z(s) = e^q E_{s' \sim p(. | s)} z(s')^{\gamma}
    """
    Q = np.diag(np.squeeze(np.exp(q)))
    return np.dot(np.dot(Q, p), z**discount)

@jit
def linear_value_functional(p, q, u, discount):
    """
    V(s) = q(s) - KL(u || p) + \gamma P_{s' \sim u} V(s')

    Args:
        p (np.ndarray): [n_states x n_states ]
        q (np.ndarray): [n_states x 1]
        u (np.ndarray): [n_states x n_states]
        discount (float): the temporal discount value
    """
    n = p.shape[0]

    r_lmdp = q + np.sum(u*np.log(p/u), axis=0)

    vs = np.dot(np.linalg.inv(np.eye(n) - discount*u), r_lmdp)
    return vs

def lmdp_solver(p, q, discount):
    """
    Solves z = QPz^a

    Args:
        p (np.ndarray): [n_states x n_states]. The unconditioned dynamics
        q (np.ndarray): [n_states x 1]. The state rewards

    Returns:
        (np.ndarray): [n_states x n_states].the optimal control
         (np.ndarray): [n_states x 1]. the value of the optimal policy
    """
    # BUG doesnt work for large discounts: 0.999.

    # Evaluate
    # Solve z = QPz
    init = np.ones((p.shape[-1], 1))
    update_fn = lambda z: linear_bellman_operator(p, q, z, discount)
    z = utils.solve(update_fn, init)[-1].squeeze()
    v = np.log(z)

    # Calculate the optimal control
    # G(x) = sum_x' p(x' | x) z(x')
    G = np.einsum('ij,i->j', p, z)
    # u*(x' | x) = p(x' | x) z(x') / G[z](x)
    u = p * z[:, np.newaxis] / G[np.newaxis, :]

    return u, v

def lmdp_decoder(u, P, lr=10):
    """
    Given optimal control dynamics.
    Optimise a softmax parameterisation of the policy.
    That yields those same dynamics.
    """
    # NOTE is there a way to solve this using linear equations?!
    # W = log(P_pi)
    # = sum_a log(P[a]) + log(pi[a])
    # M = log(u)
    # UW - UM = 0
    # U(W-M) = 0, W = M = sum_a log(P[a]) + log(pi[a])
    # 0 = sum_a log(P[a]) + log(pi[a]) - M

    def loss(pi_logits):
        pi = utils.softmax(pi_logits)
        # P_pi(s'|s) = \sum_a pi(a|s)p(s'|s, a)
        P_pi = np.einsum('ijk,jk->ij', P, pi)
        return np.sum(np.multiply(u, np.log(u/P_pi)))  # KL

    dLdw = jit(grad(loss))
    def update_fn(w):
        return w - lr * dLdw(w)

    init = rnd.standard_normal((P.shape[0], P.shape[-1]))
    pi_star_logits = utils.solve(update_fn, init)[-1]

    return utils.softmax(pi_star_logits)

def option_transition_fn(P, k):
    n_states = P.shape[0]
    Ps = [P]
    for i in range(k-1):
        P_i = np.einsum('ijk,ijl->ijkl', Ps[-1], P).reshape((n_states, n_states, -1))
        Ps.append(P_i)
    return np.concatenate(Ps, axis=-1)

def lmdp_option_decoder(u, P, lr=1, k=5):
    """
    Given optimal control dynamics.
    Optimise a softmax parameterisation of the policy.
    That yields those same dynamics.
    """
    n_states = P.shape[0]
    n_actions = P.shape[-1]

    # the augmented transition fn. [n_states, n_states, n_options]
    P_options = option_transition_fn(P, k)

    def loss(option_logits):
        options = utils.softmax(option_logits)
        # P_pi(s'|s) = \sum_w pi(w|s)p(s'|s, w)
        P_pi = np.einsum('ijk,jk->ij', P_options, options)
        return np.sum(np.multiply(u, np.log(u/P_pi)))  # KL

    dLdw = jit(grad(loss))
    def update_fn(w):
        return w - lr * dLdw(w)

    n_options = sum([n_actions**(i+1) for i in range(k)])
    print('N options: {}'.format(n_options))
    init = rnd.standard_normal((P.shape[0], n_options))
    pi_star_logits = utils.solve(update_fn, init)[-1]

    return utils.softmax(pi_star_logits)
