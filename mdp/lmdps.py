import numpy as onp
import numpy.random as rnd

import jax.numpy as np
from jax import grad, jit
from jax.scipy.special import logsumexp

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

        This ensures that p, q are able to `represent` the original dynamics and
        reward.

        See supplementary material of todorov 2009 for more info
        https://www.pnas.org/content/106/28/11478
        """
        # b_a = r(x, a)
        b = r[idx_x, :]
        # D_ax' = p(x' | x, a)
        D = P[:, idx_x, :]

        # D(q.1 - m) = b,
        # Solve for c where D c = b
        # c = q.1 - m  => m = q.1 - c
        # p = exp(m) = exp(q.1 - c) \propto exp(-c)
        # So p = softmax(-c)
        # And q is the normalization constant?
        
        # Original code:
        # c = np.dot(b, linalg.pinv(D))
        # q = np.log(np.sum(np.exp(c)))
        # m = c - q
        # p = np.exp(m)
        
        # Let's stick to the original logic but clean it up.
        # Note: linalg.pinv is numpy's pinv, need to use jax or numpy. 
        # The input P is likely numpy array based on usage in experiments.
        
        c = onp.dot(b, onp.linalg.pinv(D))
        
        # q = log sum exp c
        # This q seems to be related to the "desirability" or free energy.
        q = onp.log(onp.sum(onp.exp(c)))
        
        # m = c - q
        # p = exp(m) = exp(c) / sum(exp(c))
        # So p is softmax(c). 
        # WAIT. The original code had m = c - q.
        # If p = exp(m) = exp(c - q) = exp(c) / exp(q).
        # exp(q) = sum(exp(c)).
        # So yes, p = softmax(c).
        
        p = onp.exp(c - q)

        # p should be a distribution
        err = onp.isclose(p.sum(0), 1.0, 1e-3)
        if not err:
            print(f"Error in normalization: sum={p.sum(0)}")
            raise ValueError('p is not normalised')

        return p, onp.array([q])

    # TODO, can solve these in parallel.
    pnqs = [embed_state(i) for i in range(P.shape[0])]
    p, q = tuple([onp.stack(val, axis=1) for val in zip(*pnqs)])
    return p, onp.squeeze(q)

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
def soft_bellman_operator(p, q, v, discount):
    """
    v(s) = q(s) + log E_{s' \sim p(.|s)} exp(gamma * v(s'))
    """
    # v: [n_states]
    # p: [n_states, n_states]
    # q: [n_states]
    
    gamma_v = discount * v
    # Stable log-sum-exp:
    # log( sum_j p_ij exp(gv_j) )
    # = max_j(gv_j) + log( sum_j p_ij exp(gv_j - max) )
    
    max_val = np.max(gamma_v)
    # Subtract max for stability
    exp_term = np.exp(gamma_v - max_val)
    # Expectation
    expected_exp = np.dot(p, exp_term)
    
    return q + max_val + np.log(expected_exp + 1e-30)

def lmdp_solver(p, q, discount):
    """
    Solves for the optimal value function v and control u.
    
    Args:
        p (np.ndarray): [n_states x n_states]. The unconditioned dynamics
        q (np.ndarray): [n_states x 1]. The state rewards
        discount (float): discount factor

    Returns:
        u (np.ndarray): [n_states x n_states]. the optimal control
        v (np.ndarray): [n_states]. the value of the optimal policy
    """
    n_states = p.shape[0]
    q = q.squeeze()
    
    # Solve for v using soft value iteration (stable)
    init_v = np.zeros(n_states)
    update_fn = lambda v: soft_bellman_operator(p, q, v, discount)
    v = utils.solve(update_fn, init_v)[-1]
    
    # Calculate the optimal control u
    # u(s'|s) \propto p(s'|s) exp(gamma * v(s'))
    
    gamma_v = discount * v
    # Use stable softmax
    # logits = log(p) + gamma * v
    # But p can be 0.
    # u_ij = p_ij * exp(gamma * v_j) / sum_k (p_ik * exp(gamma * v_k))
    
    # Numerator
    num = p * np.exp(gamma_v - np.max(gamma_v)) # shift for stability
    # Denominator
    denom = np.sum(num, axis=1, keepdims=True)
    
    u = num / (denom + 1e-30)
    
    return u, v

def lmdp_decoder(u, P, lr=10):
    """
    Given optimal control dynamics.
    Optimise a softmax parameterisation of the policy.
    That yields those same dynamics.
    """
    def loss(pi_logits):
        pi = utils.softmax(pi_logits)
        # P_pi(s'|s) = \sum_a pi(a|s)p(s'|s, a)
        P_pi = np.einsum('ijk,jk->ij', P, pi)
        return np.sum(np.multiply(u, np.log(u/(P_pi+1e-30) + 1e-30)))  # KL

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
        return np.sum(np.multiply(u, np.log(u/(P_pi+1e-30) + 1e-30)))  # KL

    dLdw = jit(grad(loss))
    def update_fn(w):
        return w - lr * dLdw(w)

    n_options = sum([n_actions**(i+1) for i in range(k)])
    print('N options: {}'.format(n_options))
    init = rnd.standard_normal((P.shape[0], n_options))
    pi_star_logits = utils.solve(update_fn, init)[-1]

    return utils.softmax(pi_star_logits)

