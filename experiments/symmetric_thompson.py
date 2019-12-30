"""
Thompson sampling with a symmetric prior.
"""
import numpy as np

import mdp.utils as utils
import mdp.abstractions as abs
import mdp.search_spaces as ss

def complexity(mdp):
    """
    Complexity is measured by how many states are the same (under some measure of similarity).
    """
    # TODO This needs to be normalised! How?
    n = mdp.S
    similar_states = np.mean(np.sum((Qs[:, :, None, :] - Qs[:, None, :, :])**2, axis=3), axis=0) # |S| x |S|
    m = np.max(np.sum((similar_states>tol), axis=1))
    return n/m

def mdp_sampler(params):
    return mdp

def symmetric_sampler(params):
    """
    Could use rejection sampling!?!
    """
    # TODO how efficient is this. probs bad... need to think about.
    while not satisfied:
        mdp = mdp_sampler(params)
        if complexity(mdp) > np.random.random():
            break
    return mdp

def thompson(mdp, lr):
    """
    Can we do TD with values from different MDPs?
    """
    V_true = vmap(lambda pi: utils.value_functional(mdp.P, mdp.r, pi, mdp.discount))
    V_guess = vmap(lambda params, pi: utils.value_functional(*mdp_sampler(params), pi, mdp.discount), in_axes=(None, None, 0))
    dLdp = grad(lambda params: mse(V_true(pis), V_guess(params, pis)))

    @jit
    def update_fn(params, Q):
        m = symmetric_sampler(params)
        Q_ = utils.bellman_optimality_operator(m.P, m.r, Q, m.discount)
        params_tp1 -= lr * dLdp(params)  # done based on observations... could use model iteration!?
        Q_tp1 = Q + lr * (Q_ - Q)
        return Q_tp1, params_tp1

    return update_fn


if __name__ == "__main__":
    # np.random.seed(0)
    n_states, n_actions = 16, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = [utils.random_policy(n_states, n_actions) for _ in range(1000)]
    # pis = utils.get_deterministic_policies(n_states, n_actions)

    onoffpolicy_abstraction(mdp, pis)
