import numpy as np
import matplotlib.pyplot as plt

import mdp.lmdps as lmdps
import mdp.utils as utils
import mdp.search_spaces as search_spaces

def onehot(x, n):
    return np.eye(n)[x]

def compare_mdp_lmdp():
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.9)
    pis = utils.gen_grid_policies(7)
    vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)

    plt.figure(figsize=(16,16))
    plt.scatter(vs[:, 0], vs[:, 1], s=10, alpha=0.75)

    # solve via LMDPs
    p, q = lmdps.mdp_encoder(mdp.P, mdp.r)
    u, v = lmdps.lmdp_solver(p, q, mdp.discount)
    pi_u_star = lmdps.lmdp_decoder(u, mdp.P)

    pi_p = lmdps.lmdp_decoder(p, mdp.P)

    # solve MDP
    init = np.random.standard_normal((n_states, n_actions))
    pi_star = utils.solve(search_spaces.policy_iteration(mdp), init)[-1]
    # pi_star = onehot(np.argmax(qs, axis=1), n_actions)

    # evaluate both policies.
    v_star = utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount)
    v_u_star = utils.value_functional(mdp.P, mdp.r, pi_u_star, mdp.discount)
    v_p = utils.value_functional(mdp.P, mdp.r, pi_p, mdp.discount)

    plt.scatter(v_star[0, 0], v_star[1, 0], c='m', alpha=0.5, marker='x', label='mdp')
    plt.scatter(v_u_star[0, 0], v_u_star[1, 0], c='g', alpha=0.5, marker='x', label='lmdp')
    plt.scatter(v_p[0, 0], v_p[1, 0], c='k', marker='x', alpha=0.5, label='p')
    plt.legend()
    plt.show()

def compare_acc():
    n_states, n_actions = 2, 2


    lmdp = []
    lmdp_rnd = []
    for _ in range(10):
        mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

        # solve via LMDPs
        p, q = lmdps.mdp_encoder(mdp.P, mdp.r)
        u, v = lmdps.lmdp_solver(p, q, mdp.discount)
        pi_u_star = lmdps.lmdp_decoder(u, mdp.P)

        # solve MDP
        init = np.random.standard_normal((n_states, n_actions))
        pi_star = utils.solve(search_spaces.policy_iteration(mdp), init)[-1]

        # solve via LMDPs
        # with p set to the random dynamics
        p, q = lmdps.mdp_encoder(mdp.P, mdp.r)
        p = np.einsum('ijk,jk->ij', mdp.P, np.ones((n_states, n_actions))/n_actions)
        # q = np.max(mdp.r, axis=1, keepdims=True)
        u, v = lmdps.lmdp_solver(p, q, mdp.discount)
        pi_u_star_random = lmdps.lmdp_decoder(u, mdp.P)

        # evaluate both policies.
        v_star = utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount)
        v_u_star = utils.value_functional(mdp.P, mdp.r, pi_u_star, mdp.discount)
        v_u_star_random = utils.value_functional(mdp.P, mdp.r, pi_u_star_random, mdp.discount)

        lmdp.append(np.isclose(v_star, v_u_star, 1e-3).all())
        lmdp_rnd.append(np.isclose(v_star, v_u_star_random, 1e-3).all())

    print([np.sum(lmdp), np.sum(lmdp_rnd)])
    plt.bar(range(2), [np.sum(lmdp), np.sum(lmdp_rnd)])
    plt.show()


def lmdp_dynamics():
    pass

def lmdp_field():
    """
    For each policy.
    Calculate its dynamics, P_pi.
    Estimate the value via the LMDP.
    Plot difference under linearTD operator.
    """
    n_states, n_actions = 2, 2
    pis = utils.gen_grid_policies(11)

    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    p, q = lmdps.mdp_encoder(mdp.P, mdp.r)

    vs = []
    dvs = []
    for pi in pis:
        u = np.einsum('ijk,jk->ij', mdp.P, pi)
        v = lmdps.linear_value_functional(p, q, u, mdp.discount)
        z = np.exp(v)
        Tz = lmdps.linear_bellman_operator(p, q, z, mdp.discount)
        dv = np.log(Tz) - np.log(z)

        vs.append(v)
        dvs.append(dv)

    dvs = np.vstack(dvs)
    vs = np.vstack(vs)

    normed_dvs = utils.normalize(dvs)

    plt.figure(figsize=(16,16))
    plt.subplot(1,2,1)
    plt.title('Linearised Bellman operator')
    plt.quiver(vs[:, 0], vs[:, 1], normed_dvs[:, 0], normed_dvs[:,1], np.linalg.norm(dvs, axis=1))

    # plot bellman
    Vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)
    diff_op = lambda V: utils.bellman_optimality_operator(mdp.P, mdp.r, np.expand_dims(V, 1), mdp.discount) - np.expand_dims(V, 1)
    dVs = np.stack([np.max(diff_op(V), axis=1) for V in Vs])

    normed_dVs = utils.normalize(dVs)

    plt.subplot(1,2,2)
    plt.title('Bellman operator')
    plt.quiver(Vs[:, 0], Vs[:, 1], normed_dVs[:, 0], normed_dVs[:,1], np.linalg.norm(dVs, axis=1))

    # plt.savefig('figs/LBO_BO.png')
    plt.show()


if __name__ == "__main__":
    # compare_mdp_lmdp()
    # compare_acc()
    lmdp_field()
