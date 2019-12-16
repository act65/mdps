"""
Hyopthesis. Sarsa shouldnt work (in general) with an abstraction that only preserves the optimal policy.
Why? Because the value of the policies is not preserved meaning that ???
Convergence doesnt work anymore. The TD operator is no longer a contraction.!?

We are taking steps Q_t+1 = TQ_t. Which is a contraction. But in the abstracted space.
There is no guarantee it is still a contraction in the original MDP?!?
"""


mdps = [utils.build_random_mdp(n_states, n_actions, 0.5) for _ in range(3*3)]
pis = utils.gen_grid_policies(31)



def test(mdp, pis):
    # n x |S| x |A|
    Qs = np.stack([evaluate(pi, P, r) for pi in pis], axis=0)

    ### all policy abstraction
    # |S| x |S|
    similar_states = np.sum((Qs[:, :, None, :] - Qs[:, None, :, :])**2, axis=[0, -1])
    all_abstracted_mdp = build_state_abstraction(similar_states, P, r)
    pi_traj = utils.solve(utils.sarsa(all_abstracted_mdp, 0.01), init)


    ### optimal policy abstraction
    pi_star = utils.solve(ss.policy_iteration(all_abstracted_mdp, 0.01), np.log(init))[-1]
    idx = np.argmin(np.stack(pis) - pi_star[None, :, :])

    similar_states = np.sum((Qs[idx, :, None, :] - Qs[idx, None, :, :])**2, axis=-1)
    optimal_abstracted_mdp = build_state_abstraction(similar_states, P, r)

    pi_traj = utils.solve(utils.q_learning(optimal_abstracted_mdp, 0.01), init)
