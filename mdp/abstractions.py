import numpy as np

"""
Alternatively, we could construct the abstraction first and then lift it to two finer MDPs?!
This would mean we could do exact abstraction!?
But how to lift to ensure only the value of the optimal policy is preserved?
Or the value of all policies is preserved?
"""

def partitions(sim):
    """

    """
    print(sim)
    print(np.where(np.triu(sim)))

    idx = np.where(np.triu(sim))
    n = len(idx[0])
    parts = {i: [] for i in range(sim.shape[0] - n)}
    # for i in range(sim.shape - n):
    #     parts[i].append(i)
    #
    # for i, j in zip(*idx):
    #     print(i,j)
    raise SystemExit
    return None



def build_state_abstraction(similar_states, P, r, tol=0.1):
    """

    """
    bools = similar_states + np.eye(similar_states.shape[0]) < tol  # approximate abstraction
    parts = partitions(bools)

    if bools.sum() == 0:
        raise ValueError('No abstraction')
    # raise SystemExit
    # exact aggregation
    # construct equivalence classes
    # a list of equivalence classes?! [(0,3,4,10), (1,2,5,6), (7,8,9,11)]

    # want a way to do this in numpy!?
    # for x, x_m1 in enumerate(e_classes):
    #     for s in x_m1:
    #         abs_r[x, :] += r[s, :]/len(x_m1)
    #         for y, y_m1 in enumerate(e_classes):
    #             for s_ in y_m1:
    #                 abs_P[y, :, x] += P[s_,:,s]/len(x_m1)

    # should calculate the error of the abstraction?!

    return None


def build_option_abstraction(k, P, r):
    """
    Want to explore how the MDP's complexity changes with k step option transformations.
    """
    pass
