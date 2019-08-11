import numpy as np
import matplotlib.pyplot as plt

import src.lmdps as lmdps
import src.utils as utils
import src.search_spaces as search_spaces

def onehot(x, n):
    return np.eye(n)[x]

def compare_mdp_lmdp():
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = utils.gen_grid_policies(41)
    vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)

    plt.figure(figsize=(16,16))
    plt.scatter(vs[:, 0], vs[:, 1], s=10, alpha=0.75)

    # solve via LMDPs
    p, q = lmdps.mdp_encoder(mdp.P, mdp.r)
    u, v = lmdps.lmdp_solver(p, q, mdp.discount)
    pi_u_star = lmdps.lmdp_decoder(u, mdp.P)

    # solve MDP
    init = np.random.standard_normal((n_states, n_actions))
    qs = utils.solve(search_spaces.value_iteration(mdp, lr=0.1), init)[-1]
    pi_star = onehot(np.argmax(qs, axis=1), n_actions)

    # evaluate both policies.
    v_star = utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount)
    v_u_star = utils.value_functional(mdp.P, mdp.r, pi_u_star, mdp.discount)

    plt.scatter(v_star[0, 0], v_star[1, 0], c='m', marker='x', label='mdp')
    plt.scatter(v_u_star[0, 0], v_u_star[1, 0], c='g', marker='x', label='lmdp')
    plt.legend()
    plt.show()

def lmdp_dynamics():
    pass

if __name__ == "__main__":
    compare_mdp_lmdp()
