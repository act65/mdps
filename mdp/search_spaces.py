"""
Explore how the different search spaces effect the GD dynamics.
"""
import functools

import numpy
import jax.numpy as np
from jax import grad, jit, jacrev, vmap

import numpy.random as rnd

import mdp.utils as utils

#############################
"""
Various ways to parameterise these fns.
Want to be able to try different topologies!!
"""

def random_parameterised_matrix(n, m, d_hidden, n_hidden):
    glorot_init = lambda shape: (1/np.sqrt(shape[0] + shape[1]))*rnd.standard_normal(shape)
    cores = [glorot_init((d_hidden, d_hidden)) for _ in range(n_hidden)]
    cores = [glorot_init((n, d_hidden))] + cores + [glorot_init((d_hidden, m))]
    return cores

def combine_svd(u, s, vT):
    return np.dot(u, np.dot(np.diag(s), vT))

def random_reparameterisation(cores, i):
    assert i > 0 and i < len(cores)-1
    n = cores[i].shape[-1]
    m = cores[i+1].shape[0]
    assert n == m  # else not invertible...

    # M = rnd.standard_normal((n,m))
    # Mm1 = np.linalg.inv(M)
    # assert np.isclose(np.dot(M, Mm1), np.eye(n), atol=1e-4).all()
    # return cores[:i-1] + [np.dot(cores[i], M)] + [np.dot(Mm1, cores[i+1])] + cores[i+2:]

    u_i, s_i, vT_i = np.linalg.svd(cores[i])
    u_ip1, s_ip1, vT_ip1 = np.linalg.svd(cores[i+1])

    # X = np.dot(cores[i], cores[i+1])
    # Y = np.dot(combine_svd(u_i, s_i, np.dot(vT_i, u_ip1)), combine_svd(np.eye(n), s_ip1, vT_ip1))
    # print(np.isclose(X, Y, atol=1e-6).all())

    return (
        cores[:i] +
        [combine_svd(u_i, s_i, np.dot(vT_i, u_ip1))] +
        [combine_svd(np.eye(n), s_ip1, vT_ip1)] +
        cores[i+2:]
        )

def build(cores):
    return functools.reduce(np.dot, cores)

######################
# Value iteration
######################

"""
Value iteration;
- Q_t+1 = Q_t + lr . (TQ_t - Q)
- and a parameterised version. where Q is a fn of some params.
"""

def sarsa(mdp, lr):
    pi = lambda Q: utils.onehot(np.argmax(Q, axis=1), Q.shape[1])  # greedy
    T = lambda Q: utils.bellman_operator(mdp.P, mdp.r, np.einsum('jk,jk->j', Q, pi(Q)), mdp.discount)
    U = lambda Q: Q + lr * (T(Q) - Q)
    return jit(U)

def q_learning(mdp, lr):
    pi = lambda Q: utils.onehot(np.argmax(Q, axis=1), Q.shape[1])  # greedy
    T = lambda Q: utils.bellman_operator(mdp.P, mdp.r, np.max(Q, axis=1), mdp.discount)
    U = lambda Q: Q + lr * (T(Q) - Q)
    return jit(U)

def value_iteration(mdp, lr):  # aka q learning
    def T(V):
        assert V.shape[1] == 1
        return utils.bellman_operator(mdp.P, mdp.r, V, mdp.discount)
    U = lambda V: V + lr * (np.max(T(V), axis=1, keepdims=True) - V)
    return jit(U)

def thompson_abstraction_value_iteration(mdp, lr):
    """
    Should be able to compare convergence speen versus vanilla value iteration.
    For problems with symmetry, expect to converge faster.
    """
    def T(V):  # current value estimate, similarity measure
        assert V.shape[1] == 1
        return np.max(utils.bellman_operator(mdp.P, mdp.r, V, mdp.discount), axis=1, keepdims=True)

    @jit
    def update_fn(X):
        V, S = X
        # X \in [0,1]^{nS x nS}. X[i, j] = 1 if V(s_i) ~= V(s_j).
        X = sample_using_symmetric_prior(S)
        # sample an abstraction / grouping (with preference towards 'symmetric' abstractions ).
        # use to update.
        return V + lr * np.dot(X, (T(V) - V))
    return update_fn

def parameterised_value_iteration(mdp, lr):
    T = lambda Q: utils.bellman_optimality_operator(mdp.P, mdp.r, Q, mdp.discount)
    TD = lambda cores: T(build(cores)) - build(cores)
    dVdw = jacrev(build)

    @jit
    def update_fn(cores):
        delta = TD(cores)
        grads = [np.einsum('ij,ijkl->kl', delta, dc) for dc in dVdw(cores)]
        # TODO attempt to understand the properties of dc. and its relation to K
        return [c+lr*g for c, g in zip(cores, grads)]
    return jit(update_fn)

def complex_value_iteration(mdp, lr):
    T = lambda Q: utils.bellman_optimality_operator(mdp.P, mdp.r, Q, mdp.discount)
    L = lambda Q: np.sum((T(np.abs(Q)) - np.abs(Q))**2)
    dLdw = grad(L)
    U = lambda Q: Q - lr * dLdw(Q)
    return jit(U)

######################
# Policy iteration
######################

def policy_iteration(mdp):
    def update_fn(pi):
        V = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)
        Q = utils.bellman_operator(mdp.P, mdp.r, V, mdp.discount)
        return utils.onehot(np.argmax(Q, axis=1), mdp.A)  # greedy update
    return update_fn

def policy_gradient_iteration_logits(mdp, lr):
    # this doesnt seem to behave nicely in larger state spaces!?
    # d/dlogits V = E_{\pi}[V] = E[V . d/dlogit log \pi]
    # dlogpi_dlogit = jacrev(lambda logits: np.log(utils.softmax(logits)+1e-8))
    dHdlogit = grad(lambda logits: utils.entropy(utils.softmax(logits)))
    dVdlogit = grad(lambda logits: np.sum(utils.value_functional(mdp.P, mdp.r, utils.softmax(logits), mdp.discount)))

    @jit
    def update_fn(logits):
        # NOTE this is actually soft A2C.
        # V = utils.value_functional(mdp.P, mdp.r, utils.softmax(logits), mdp.discount)
        # Q = utils.bellman_operator(mdp.P, mdp.r, V, mdp.discount)
        # A = Q-V
        # g = np.einsum('ijkl,ij->kl', dlogpi_dlogit(logits), A)

        g = dVdlogit(logits)

        return logits + lr * utils.clip_by_norm(g, 500) + 1e-8*dHdlogit(logits)
    return update_fn

def parameterised_policy_gradient_iteration(mdp, lr):
    dlogpi_dw = jacrev(lambda cores: np.log(utils.softmax(build(cores), axis=1)+1e-8))
    dHdw = jacrev(lambda cores: utils.entropy(utils.softmax(build(cores))))

    @jit
    def update_fn(cores):
        V = utils.value_functional(mdp.P, mdp.r, utils.softmax(build(cores), axis=1), mdp.discount)
        Q = utils.bellman_operator(mdp.P, mdp.r, V, mdp.discount)
        A = Q-V
        grads = [np.einsum('ijkl,ij->kl', d, A) for d in dlogpi_dw(cores)]
        return [c+lr*utils.clip_by_norm(g, 100)+1e-6*dH for c, g, dH in zip(cores, grads, dHdw(cores))]
    return update_fn

######################
# Model iteration
######################

def parse_model_params(n_states, n_actions, params):
    # |S| x |S| x |A| + |S| x |A|
    n = n_states * n_states * n_actions
    return params[:n].reshape((n_states, n_states, n_actions)), params[n:].reshape((n_states, n_actions))

def model_iteration(mdp, lr, pis):
    V_true = vmap(lambda pi: utils.value_functional(mdp.P, mdp.r, pi, mdp.discount))
    V_guess = vmap(lambda P, r, pi: utils.value_functional(P, r, pi, mdp.discount), in_axes=(None, None, 0))


    def loss_fn(params):
        p_logits, r = parse_model_params(mdp.S, mdp.A, params)
        return np.sum((V_true(pis) - V_guess(utils.softmax(p_logits), r, pis))**2)

    dLdp = grad(loss_fn)

    @jit
    def update_fn(params):
        return params - lr*utils.clip_by_norm(dLdp(params), 100)

    return update_fn

######################

def momentum_bundler(update_fn, decay):
    """
    Wraps an update fn in with its exponentially averaged grad.
    Uses the exponentially averaged grad to make updates rather than the grad itself.

    Args:
        U (callable): The update fn. U: params-> new_params.
        decay (float): the amount of exponential decay

    Returns:
        (callable): The new update fn. U: params'-> new_params'. Where params' = [params, param_momentum].
    """
    def momentum_update_fn(x):
        W_t, M_t = x[0], x[1]

        # TODO want a nicer way to do thisself.
        # nested fn application
        if isinstance(W_t, np.ndarray):
            dW = update_fn(W_t) - W_t  # should divide by lr here?
            M_tp1 = decay * M_t + dW
            W_tp1 = W_t + (1 - decay) * M_tp1
        elif isinstance(W_t, list):
            dW = [w_tp1 - w_t for w_tp1, w_t in zip(update_fn(W_t), W_t)]
            M_tp1 = [decay * m_t + dw for m_t, dw in zip(M_t, dW)]
            W_tp1 = [w_t + (1 - decay) * m_tp1 for w_t, m_tp1 in zip(W_t, M_tp1)]
        else:
            raise ValueError('Unknown format: {}'.format(type(W_t)))

        return W_tp1, M_tp1
    return jit(momentum_update_fn)

def approximate(v, cores, lr=1e-2, activation_fn=lambda x: x):
    """
    cores = random_parameterised_matrix(2, 1, d_hidden=8, n_hidden=4)
    v = rnd.standard_normal((2,1))
    cores_ = approximate(v, cores)
    print(v, '\n',build(cores_))
    """
    loss = lambda cores: np.sum(np.square(v - activation_fn(build(cores))))
    dl2dc = grad(loss)
    l2_update_fn = lambda cores: [c - lr*g for g, c in zip(dl2dc(cores), cores)]
    init = (cores, [np.zeros_like(c) for c in cores])
    final_variables, momentum_var = utils.solve(momentum_bundler(l2_update_fn, 0.9), init)[-1]
    return final_variables


if __name__ == '__main__':
    n_states, n_actions = 2, 2

    # # TEST parameterised
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(build(C).shape)
    #
    # # but these wont be distributed unformly in policy space!?
    # C = random_parameterised_matrix(n_states, n_actions, 8, 2)
    # print(pi(C).shape)


    mdp = build_random_mdp(n_states, n_actions, 0.9)
    init = rnd.standard_normal((n_states, n_actions))
    init /= init.sum(axis=1, keepdims=True)
    # pis = solve(policy_gradient_iteration(mdp, 0.01), init)
    # print(pis)

    # V = lambda pi: value_functional(mdp.P, mdp.r.reshape((-1, 1)), mpi(init), mdp.discount)
    # delta = lambda pi: (1/(pi*np.log(mdp.A)))  # dpi . dlog pi
    # print(delta(init).shape, V(init).shape)

    # U = policy_gradient_iteration(mdp, 0.01)
    # print(U(init))


    # x = np.abs(rnd.standard_normal((5,5)))
    # x = clip_by_norm(x, axis=1, norm=1)
    # print(x.sum(axis=1))
