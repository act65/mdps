import numpy as np

import mdp.utils as utils
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

def construct_abstraction_fn(mapping, parts, idx):
    m = len(idx)
    d = len(mapping.keys())
    f = np.zeros((m, d))
    # print(mapping)
    for k, v in mapping.items():
        i = parts.index(v)
        f[i, k] += 1
    assert not (f > 1).any()
    return f

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
    print('Abstracting from {} states to {} states'.format(mdp.S, len(parts)))
    idx = list(set(np.array([p[0] for p in parts])))  # this might cause problems.!?
    print(mapping)
    mapping, parts = fix_mapping(mapping)
    print(mapping)

    f = construct_abstraction_fn(mapping, parts, idx)
    abs_mdp = abstract_the_mdp(mdp, idx)
    print(parts)

    # want a way to do this stuff in numpy!?
    # should calculate the error of the abstraction?! check it is related to tol!?

    return abs_mdp, f, idx


def build_option_abstraction(k, P, r):
    """
    Want to explore how the MDP's complexity changes with k step option transformations.
    """
    pass


if __name__ == '__main__':
    n_states, n_actions = 16, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = [utils.random_policy(n_states, n_actions) for _ in range(100)]
    Qs = np.stack([utils.value_functional(mdp.P, mdp.r, pi, mdp.discount) for pi in pis], axis=0)

    similar_states = np.mean(np.sum((Qs[:, :, None, :] - Qs[:, None, :, :])**2, axis=3), axis=0) # |S| x |S|
    abstracted_mdp, f, idx = build_state_abstraction(similar_states, mdp)
    print(idx, len(idx), f.shape)
