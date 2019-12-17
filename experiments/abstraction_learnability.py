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
    init = np.random.standard_normal((mdp.S, mdp.A))

    # n x |S| x |A|
    Qs = np.stack([utils.value_functional(mdp.P, mdp.r, pi, mdp.discount) for pi in pis], axis=0)

    ### all policy abstraction
    similar_states = np.mean(np.sum((Qs[:, :, None, :] - Qs[:, None, :, :])**2, axis=3), axis=0) # |S| x |S|
    all_abstracted_mdp = abs.build_state_abstraction(similar_states, mdp.P, mdp.r)
    # pi_traj = utils.solve(utils.sarsa(all_abstracted_mdp, 0.01), init)

    ### optimal policy abstraction
    pi_star = utils.solve(ss.policy_iteration(mdp), np.log(init))[-1]
    idx = np.argmin(np.sum((np.stack(pis) - pi_star[None, :, :])**2, axis=(1,2)), axis=0)

    similar_states = np.sum((Qs[idx, :, None, :] - Qs[idx, None, :, :])**2, axis=-1)  # |S| x |S|
    optimal_abstracted_mdp = abs.build_state_abstraction(similar_states, mdp.P, mdp.r)

    # pi_traj = utils.solve(utils.q_learning(optimal_abstracted_mdp, 0.01), init)


if __name__ == "__main__":
    # np.random.seed(0)
    n_states, n_actions = 12, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = [utils.random_policy(n_states, n_actions) for _ in range(20)]

    onoffpolicy_abstraction(mdp, pis)
