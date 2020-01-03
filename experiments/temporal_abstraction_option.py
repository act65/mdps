import numpy as np

import mdp.utils as utils

def k_step_option_similarity():
    n_states, n_actions = 6, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    pi = utils.random_policy(n_states, n_actions)
    P = multi_step_transition_fn(mdp.P, pi, 3)
    # P[:,-1] = P[:,-2]
    # s(o1, o2) = sum_s' P(s' | s1) * log( P(s' | s2)  /  P(s' | s1))
    kl = -np.sum(P[:, :, None] * np.log(P[:, None, :]/P[:, :, None]), axis=0)
    print(kl)


if __name__ == '__main__':
    k_step_state_similarity()
