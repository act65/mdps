import jax.numpy as np

######################
# Neural tangent kernel and ...
######################

"""
Inspired by Towards Characterizing Divergence in Deep Q-Learning
https://arxiv.org/abs/1903.08894
"""

def adjusted_value_iteration(mdp, lr, D, K):
    T = lambda Q: utils.bellman_optimality_operator(mdp.P, mdp.r, Q, mdp.discount)
    U = lambda Q: Q + lr * np.dot(K, np.dot(D, T(Q) - Q))
    return jit(U)

# def corrected_value_iteration(mdp, lr):
#     T = lambda theta: mdp.r + mdp.discount * np.argmax(mdp.P * Q(w))
#     dQdw = lambda w: grad(Q)
#     Km1 = lambda w: np.linalg.inv(np.dot(dQdw(w).T, dQdw(w)))
#     U = lambda w: w + lr * np.dot(dQdw(w), np.dot(Km1, T(Q(w) - Q(w))))
#     return U
