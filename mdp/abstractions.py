import numpy as np

def build_state_abstraction(similar_states, P, r):
    # exact aggregation
    np.where(similar_states == 0)
    # construct equivalence classes
    # a list of equivalence classes?! [(0,3,4,10), (1,2,5,6), (7,8,9,11)]

    # want a way to do this in numpy!?
    for x, x_m1 in enumerate(e_classes):
        for s in x_m1:
            abs_r[x, :] += r[s, :]/len(x_m1)
            for y, y_m1 in enumerate(e_classes):
                for s_ in y_m1:
                    abs_P[y, :, x] += P[s_,:,s]/len(x_m1)

    # should calculate the error of the abstraction?!

    return loss


def build_option_abstraction(k, P, r):
    """
    Want to explore how the MDP's complexity changes with k step option transformations.
    """
    pass
