import numpy as np

import mdp.utils as utils
import mdp.search_spaces as ss
"""
Alternatively, we could construct the abstraction first and then lift it to two finer MDPs?!
This would mean we could do exact abstraction!?
But how to lift to ensure only the value of the optimal policy is preserved?
Or the value of all policies is preserved?
"""

#### TODO write tests!!!

def partitions(sim):
    """

    """
    d = sim.shape[0]
    pairs = np.where(np.triu(sim))
    mapping = {k: [] for k in range(d)}
    for i, j in zip(*pairs):
        mapping[i].append(j)
        mapping[j].append(i)

    mapping = {k: tuple(sorted([k] + v)) for k, v in mapping.items()}
    parts = list(set(mapping.values())) # this doesnt work when some similarities are not transitive!?
    return mapping, parts

def construct_abstraction_fn(mapping, idx, m, d):
    """
    Construct a mapping to lift abstracted solutions back to the ground problem
    """
    f = np.zeros((m, d))  # n_originial, n_abstract
    # print('\n\n')
    # print(mapping, parts, idx)
    # raise SystemExit
    for k, v in mapping.items():
        i = idx.index(v[0])
        f[k, i] += 1
    assert not (f > 1).any()
    return f.T

def abstract_the_mdp(mdp, idx):

    abs_P = mdp.P[idx, :, :][:, idx, :]
    abs_r = mdp.r[idx, :]

    return utils.MDP(len(idx), mdp.A, abs_P, abs_r, mdp.discount, mdp.d0)

def shared(x, y):
    return any(i in y for i in x)

def fix_mapping(mapping):
    n = len(mapping.keys())
    new_mapping = {i: [] for i in range(n)}
    for k1, v1 in mapping.items():
        for k2, v2 in mapping.items():
            if shared(v1, v2):
                new_mapping[k1] += list(set(v1).union(set(v2)))

    new_mapping = {k: tuple(set(v)) for k ,v in new_mapping.items()}
    return new_mapping, list(set(new_mapping.values()))


def build_state_abstraction(similar_states, mdp, tol=0.1):
    """

    """
    bools = similar_states + np.eye(similar_states.shape[0]) < tol  # approximate abstraction

    if bools.sum() == 0:
        raise ValueError('No abstraction')

    mapping, parts = partitions(bools)
    idx = list(set(np.array([p[0] for p in parts])))  # pick a representative set of states. one from each partition
    f = construct_abstraction_fn(mapping, idx, mdp.S, len(idx))

    # print('Abstracting from {} states to {} states'.format(mdp.S, len(parts)))
    # print('idx', idx)
    # print('mapping', mapping)
    # print('parts', parts)
    # mapping, parts = fix_mapping(mapping)
    # print(f)
    # print(f.shape, abs_mdp.S)

    abs_mdp = abstract_the_mdp(mdp, idx)

    # want a way to do this stuff in numpy!?
    # should calculate the error of the abstraction?! check it is related to tol!?

    return idx, abs_mdp, f


def build_option_abstraction(k, P, r):
    """
    Want to explore how the MDP's complexity changes with k step option transformations.
    """
    pass


def PI(init, M, f):
    pi_star = utils.solve(ss.policy_iteration(M), np.log(init))[-1]
    return utils.value_functional(M.P, M.r, np.dot(f.T, pi_star), M.discount)

def Q(init, M, f):

    # solve
    V_init = utils.value_functional(M.P, M.r, init, M.discount)
    Q_init = utils.bellman_operator(M.P, M.r, V_init, M.discount)
    Q_star = utils.solve(ss.q_learning(M, 0.01), Q_init)[-1]
    # lift
    return np.dot(f.T, np.max(Q_star, axis=1, keepdims=True))

def SARSA(init, M, f):

    # solve
    V_init = utils.value_functional(M.P, M.r, init, M.discount)
    Q_init = utils.bellman_operator(M.P, M.r, V_init, M.discount)
    Q_star = utils.solve(ss.sarsa(M, 0.01), Q_init)[-1]

    # lift
    return np.dot(f.T, np.max(Q_star, axis=1, keepdims=True))

if __name__ == '__main__':

    tol = 0.01

    n_states, n_actions = 512, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    init = np.random.random((mdp.S, mdp.A))
    init = init / np.sum(init, axis=1, keepdims=True)
    pi_star = utils.solve(ss.policy_iteration(mdp), np.log(init))[-1]

    Q_star = utils.bellman_operator(mdp.P, mdp.r,utils.value_functional(mdp.P, mdp.r, pi_star, mdp.discount), mdp.discount)


    similar_states = np.sum(np.abs(Q_star[:, None, :] - Q_star[None, :, :]), axis=-1)  # |S| x |S|
    optimal_idx, optimal_abstracted_mdp, optimal_f = build_state_abstraction(similar_states, mdp, tol)

    truth = PI(init, mdp, np.eye(mdp.S))
    approx = Q(init[optimal_idx], optimal_abstracted_mdp, optimal_f)

    print('\n', 'bound >=V*-V', '\n', '{} >= {}'.format(2*tol/(1-mdp.discount)**2, np.max(np.abs(truth - approx))))
