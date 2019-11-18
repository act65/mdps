import collections
import itertools


import jax.numpy as np
from jax import jit
import numpy.random as rnd
import numpy

import mdp.search_spaces as search_spaces

def onehot(x, N):
    return np.eye(N)[x]

def entropy(p):
    return -np.sum(np.log(p+1e-8) * p)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x, axis=-1):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def normalize(x):
    mags = np.linalg.norm(x, axis=1, keepdims=True)
    return x/mags

def clip_by_norm(x, norm):
    v = np.linalg.norm(x)
    p = sigmoid(v - norm)
    # p * corrected vs 1-p * uncorrected
    return p * x * norm / v + (1-p) * x

MDP = collections.namedtuple('mdp', ['S', 'A', 'P', 'r', 'discount', 'd0'])

def build_random_mdp(n_states, n_actions, discount):
    P = rnd.random((n_states, n_states, n_actions))
    r = rnd.standard_normal((n_states, n_actions))
    d0 = rnd.random((n_states, 1))
    return MDP(n_states, n_actions, P/P.sum(axis=0, keepdims=True), r, discount, d0/d0.sum(axis=0, keepdims=True))

######################

def gen_grid_policies(N):
    # special case for 2 x 2
    p1s, p2s = np.linspace(0,1,N), np.linspace(0,1,N)
    p1s = p1s.ravel()
    p2s = p2s.ravel()
    return [np.array([[p1, 1-p1],[1-p2, p2]]) for p1 in p1s for p2 in p2s]

# def gen_grid_policies(N, n_states=2, n_actions=2):
#     # special case for 2 x 2
#     p1s, p2s = np.linspace(0,1,N), np.linspace(0,1,N)
#     p1s = p1s.ravel()
#     p2s = p2s.ravel()
#     return [np.array([[p1, 1-p1],[1-p2, p2]]) for p1 in p1s for p2 in p2s]

def get_deterministic_policies(n_states, n_actions):
    simplicies = list([np.eye(n_actions)[i] for i in range(n_actions)])
    pis = list(itertools.product(*[simplicies for _ in range(n_states)]))
    return [np.stack(p) for p in pis]

def get_random_policy_2x2():
    p1 = rnd.random()
    p2 = rnd.random()
    return np.array([[p1, 1-p1], [p2, 1-p2]])

def rnd_simplex(d):
    pts = rnd.uniform(0, 1, d-1)
    return numpy.diff([0]+ sorted(pts) + [1])

def random_policy(n_states, n_actions):
    return np.vstack([rnd_simplex(n_actions) for _ in range(n_states)])

def random_det_policy(n_states, n_actions):
    return np.vstack([onehot(rnd.randint(0, n_actions), n_actions) for _ in range(n_states)])

@jit
def polytope(P, r, discount, pis):
    print('n pis:{}'.format(len(pis)))
    def V(pi):
        return np.sum(value_functional(P, r, pi, discount), axis=1)
    vs = np.vstack([V(pi) for pi in pis])
    return vs

"""
Some useful functions that will be repeately used.
- `value_functional`: evaluates a policy within a mdp
- `bellman_optimality_operator`: calculates a step of the bellman operator
"""

@jit
def value_functional(P, r, pi, discount):
    """
    V = r_{\pi} + \gamma P_{\pi} V
      = (I-\gamma P_{\pi})^{-1}r_{\pi}

    Args:
        P (np.ndarray): [n_states x n_states x n_actions]
        r (np.ndarray): [n_states x n_actions]
        pi (np.ndarray): [n_states x n_actions]
        discount (float): the temporal discount value
    """
    n = P.shape[0]
    # P_{\pi}(s_t+1 | s_t) = sum_{a_t} P(s_{t+1} | s_t, a_t)\pi(a_t | s_t)
    P_pi = np.einsum('ijk,jk->ij', P, pi)
    r_pi = np.expand_dims(np.einsum('ij,ij->i', pi, r), 1)

    # assert np.isclose(pi/pi.sum(axis=1, keepdims=True), pi).all()
    # assert np.isclose(P_pi/P_pi.sum(axis=0, keepdims=True), P_pi, atol=1e-4).all()

    # BUG why transpose here?!?!
    vs = np.dot(np.linalg.inv(np.eye(n) - discount*P_pi.T), r_pi)
    # print(vs.shape, P_pi.shape)
    return vs

def bellman_optimality_operator(P, r, Q, discount):
    """
    Args:
        P (np.ndarray): [n_states x n_states x n_actions]
        r (np.ndarray): [n_states x n_actions]
        Q (np.ndarray): [n_states x n_actions]
        discount (float): the temporal discount value

    Returns:
        (np.ndarray): [n_states, n_actions]
    """
    if len(Q.shape) == 1:
        Q = np.expand_dims(Q, 1)

    if Q.shape[1] == 1:  # Q == V
        # Q(s, a) =  r(s, a) + \gamma E_{s'~P(s' | s, a)} V(s')
        return r + discount*np.einsum('ijk,il->jk', P, Q)
    else:
        # Q(s, a) =  r(s, a) + \gamma max_a' E_{s'~P(s' | s, a)} Q(s', a')
        return r + discount*np.max(np.einsum('ijk,il->jkl', P, Q), axis=-1)

"""
Tools for simulating dyanmical systems.
"""

def isclose(x, y, atol=1e-8):
    if isinstance(x, np.ndarray):
        return np.isclose(x, y, atol=atol).all()
    elif isinstance(x, list):
        # return all(np.isclose(x[0], y[0], atol=1e-03).all() for i in range(len(x)))
        return np.isclose(search_spaces.build(x), search_spaces.build(y), atol=atol).all()
    elif isinstance(x, tuple) and isinstance(x[0], np.ndarray):
        return np.isclose(x[0], y[0], atol=atol).all()
    elif isinstance(x, tuple) and isinstance(x[0], list):
        return np.isclose(search_spaces.build(x[0]), search_spaces.build(y[0]), atol=atol).all()
    else:
        raise ValueError('wrong format')

def converged(l):
    if len(l)>1:
        if len(l)>5000 and isclose(l[-1], l[-2], 1e-6):
            return True
        if len(l)>10000 and isclose(l[-1], l[-2], 1e-4):
            return True
        if isclose(l[-1], l[-2], 1e-8):
            return True
        if len(l)>20000:
            print(l[-5:-1])
            raise ValueError('not converged...')
        # if np.isnan(l[-1]).any():
        #     raise ValueError('NaNs')
    return False

def solve(update_fn, init):
    xs = [init]
    x = init
    while not converged(xs):
        x = update_fn(x)
        xs.append(x)
        print('\rStep: {}'.format(len(xs)), end='', flush=True)
    return xs
