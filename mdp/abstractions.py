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

def construct_abstraction_fn(mapping, parts):
    m = len(parts)
    d = len(mapping.keys())
    f = np.zeros((m, d))
    for k, v in mapping.items():
        i = parts.index(v)
        f[i, k] += 1
    return f

def abstract_the_mdp(mdp, parts):
    idx = np.array([p[0] for p in parts])

    abs_P = mdp.P[idx, :, :][:, idx, :]
    abs_r = mdp.r[idx, :]

    return utils.MDP(len(parts), mdp.A, abs_P, abs_r, mdp.discount, mdp.d0)


def build_state_abstraction(similar_states, mdp, tol=0.1):
    """

    """
    bools = similar_states + np.eye(similar_states.shape[0]) < tol  # approximate abstraction

    if bools.sum() == 0:
        raise ValueError('No abstraction')

    mapping, parts = partitions(bools)
    print('Abstracting from {} states to {} states'.format(mdp.S, len(parts)))
    f = construct_abstraction_fn(mapping, parts)
    abs_mdp = abstract_the_mdp(mdp, parts)

    # want a way to do this stuff in numpy!?
    # should calculate the error of the abstraction?! check it is related to tol!?

    return abs_mdp, f


def build_option_abstraction(k, P, r):
    """
    Want to explore how the MDP's complexity changes with k step option transformations.
    """
    pass
