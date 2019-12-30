"""
Hyopthesis. Sarsa shouldnt work (in general) with an abstraction that only preserves the optimal policy.
Why? Because the value of the policies is not preserved meaning that ???
Convergence doesnt work anymore. The TD operator is no longer a contraction.!?

We are taking steps Q_t+1 = TQ_t. Which is a contraction. But in the abstracted space.
There is no guarantee it is still a contraction in the original MDP?!?
"""

import numpy as np

import mdp.utils as utils
import mdp.abstractions as abs
import mdp.search_spaces as ss



def onoffpolicy_abstraction(mdp, pis):

    def PI(init, M, f):
        pi_star = utils.solve(ss.policy_iteration(M), np.log(init))[-1]
        return utils.value_functional(mdp.P, mdp.r, np.dot(f.T, pi_star), mdp.discount)

    def VI(init, M, f):
        V_init = utils.value_functional(M.P, M.r, init, M.discount)
        Q_init = utils.bellman_optimality_operator(M.P, M.r, V_init, M.discount)
        pi_star = utils.solve(ss.value_iteration(M, 0.01), Q_init)[-1]
        return utils.value_functional(mdp.P, mdp.r, np.dot(f.T, pi_star), mdp.discount)

    def SARSA(init, M, f):
        V_init = utils.value_functional(M.P, M.r, init, mdp.discount)
        pi_star = utils.solve(ss.sarsa(M, 0.01), V_init)[-1]
        return utils.value_functional(mdp.P, mdp.r, np.dot(f.T, pi_star), mdp.discount)

    init = np.random.random((mdp.S, mdp.A))
    init = init / np.sum(init, axis=1, keepdims=True)

    # n x |S| x |A|
    Qs = np.stack([utils.value_functional(mdp.P, mdp.r, pi, mdp.discount) for pi in pis], axis=0)

    ### all policy abstraction
    similar_states = np.mean(np.sum((Qs[:, :, None, :] - Qs[:, None, :, :])**2, axis=3), axis=0) # |S| x |S|
    all_abstracted_mdp, all_f, all_idx = abs.build_state_abstraction(similar_states, mdp)
    # pi_traj = utils.solve(ss.sarsa(all_abstracted_mdp, 0.01), np.sum(init, axis=1, keepdims=True))
    # pi_traj = utils.solve(ss.sarsa(all_abstracted_mdp, 0.01), np.sum(init, axis=1, keepdims=True))
    # all_abstract_pi_traj = utils.solve(ss.policy_iteration(all_abstracted_mdp), np.log(init[:all_abstracted_mdp.S, :all_abstracted_mdp.A]))

    ### optimal policy abstraction
    pi_star = utils.solve(ss.policy_iteration(mdp), np.log(init))[-1]
    idx = np.argmin(np.sum((np.stack(pis) - pi_star[None, :, :])**2, axis=(1,2)), axis=0)

    similar_states = np.sum((Qs[idx, :, None, :] - Qs[idx, None, :, :])**2, axis=-1)  # |S| x |S|
    optimal_abstracted_mdp, optimal_f, optimal_idx = abs.build_state_abstraction(similar_states, mdp)
    # pi_traj = utils.solve(utils.q_learning(optimal_abstracted_mdp, 0.01), init)
    # optimal_abstract_pi_traj = utils.solve(ss.policy_iteration(optimal_abstracted_mdp), np.log(init[:optimal_abstracted_mdp.S, :optimal_abstracted_mdp.A]))

    mdps = [mdp, all_abstracted_mdp, optimal_abstracted_mdp]
    names = ['ground', 'all_abstracted_mdp', 'optimal_abstracted_mdp']
    solvers = [PI, VI, SARSA]
    lifts = [np.eye(mdp.S), all_f, optimal_f]
    idxs = [range(mdp.S), all_idx, optimal_idx]
    print(idxs)
    # solver_names = ['Pi']
    results = [(n, solve.__name__, solve(init[idx, :], M, f)) for M, n, f, idx in zip(mdps, names, lifts, idxs) for solve in solvers]
    print(results)

    # # now compare them.
    # names = ['policy iteration', 'all_abstracted_mdp', 'optimal_abstracted_mdp', 'rnd']
    # # lift the policies back to the original mdp.
    # pis = [pi_star, np.dot(all_f.T, all_abstract_pi_traj[-1]), np.dot(optimal_f.T, optimal_abstract_pi_traj[-1]), utils.random_policy(mdp.S, mdp.A)]
    # Vs = []
    # for name, pi in zip(names, pis):
    #     V = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)
    #     Vs.append(V)
    #     print(V)
    # print(np.array([[np.sum((i-j)**2) for i in Vs] for j in Vs]))



if __name__ == "__main__":
    # np.random.seed(0)
    n_states, n_actions = 16, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = [utils.random_policy(n_states, n_actions) for _ in range(1000)]
    # pis = utils.get_deterministic_policies(n_states, n_actions)

    onoffpolicy_abstraction(mdp, pis)
