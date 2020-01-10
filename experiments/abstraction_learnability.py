"""
Hyopthesis.
Sarsa shouldnt work (in general) with an abstraction that only preserves the optimal policy.
Why? Because the value of the policies is not preserved meaning that
T(Q)-Q is not guaranteed to point in sensible directions.

Most of the time, we might be ok tho?

We are taking steps Q_t+1 = TQ_t. Which is a contraction. But in the abstracted space.
There is no guarantee it is still a contraction in the original MDP?!?
"""

import numpy as np

import mdp.utils as utils
import mdp.abstractions as abs
import mdp.search_spaces as ss



def onoffpolicy_abstraction(mdp, pis):
    tol = 0.01

    init = np.random.random((mdp.S, mdp.A))
    init = init / np.sum(init, axis=1, keepdims=True)


    # ### all policy abstraction
    # # n x |S| x |A|
    # Qs = np.stack([utils.bellman_operator(mdp.P, mdp.r, utils.value_functional(mdp.P, mdp.r, pi, mdp.discount), mdp.discount) for pi in pis], axis=0)
    # similar_states = np.sum(np.sum(np.abs(Qs[:, :, None, :] - Qs[:, None, :, :]), axis=3), axis=0) # |S| x |S|
    # all_idx, all_abstracted_mdp, all_f = abs.build_state_abstraction(similar_states, mdp)

    ### optimal policy abstraction
    pi_star = utils.solve(ss.policy_iteration(mdp), np.log(init))[-1]
    Q_star = utils.bellman_operator(mdp.P, mdp.r,utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount), mdp.discount)

    # similar_states = np.sum(np.abs(Q_star[:, None, :] - Q_star[None, :, :]), axis=-1)  # |S| x |S|. preserves optimal policy's value (for all actions)
    # similar_states = np.abs(np.max(Q_star[:, None, :],axis=-1) - np.max(Q_star[None, :, :],axis=-1))  # |S| x |S|. preserves optimal action's value

    #
    V = utils.value_functional(mdp.P, mdp.r, init, mdp.discount)
    similar_states = np.abs(V[None, :, :]-V[:, None, :])[:, :, 0]

    optimal_idx, optimal_abstracted_mdp, optimal_f = abs.build_state_abstraction(similar_states, mdp, tol)

    mdps = [mdp, optimal_abstracted_mdp]
    names = ['ground', 'optimal_abstracted_mdp']
    solvers = [abs.Q, abs.SARSA, abs.VI]
    lifts = [np.eye(mdp.S), optimal_f]
    idxs = [range(mdp.S), optimal_idx]

    # if all_f.shape[0] == optimal_f.shape[0]:
    #     raise ValueError('Abstractions are the same so we probs wont see any difference...')
    print('\nAbstraction:', optimal_f.shape)

    truth = abs.PI(init, mdp, np.eye(mdp.S))
    results = []
    for n, M, idx, f in zip(names, mdps, idxs, lifts):
        for solve in solvers:
            err = np.max(np.abs(truth - solve(init[idx, :], M, f)))
            results.append((n, solve.__name__, err))
    return results



if __name__ == "__main__":
    # np.random.seed(0)
    n_states, n_actions = 512, 2
    pis = [utils.random_policy(n_states, n_actions) for _ in range(10)]
    # pis = utils.get_deterministic_policies(n_states, n_actions)

    for _ in range(10):
        mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
        results = onoffpolicy_abstraction(mdp, pis)

        n = len(results)
        print('\n')
        for i in range(n//2):
            print(results[n//2+i][-2], results[n//2+i][-1])
