import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from polytope_tools import *

import collections

def discount_partitions():
    """
    Plot the partitions of discounts wrt optimal policies.

    Ok, so a MDP that has many changes of the optimal policy wrt to discounts is
    a 'harder' one than one that has few changes.
    Can solve once / few times and transfer the policy between different discounts.
    When / why can this be done?
    """
    n_states, n_actions = 3, 3
    n = 100
    discounts = np.linspace(0.1, 0.999, n)


    for _ in range(10):
        P, r = generate_rnd_problem(n_states, n_actions)
        stars = []
        for i in range(n):
            stars.append(solve(P, r, discounts[i]))

        diffs = [np.sum(np.abs(stars[i]-stars[i+1])) for i in range(n-1)]
        plt.plot(discounts[:-1], np.cumsum(diffs))

    plt.title('Discount partitions')
    plt.xlabel('Discount')
    plt.ylabel('Cumulative changes to the optimal policy')

    # plt.show()
    plt.savefig('../pictures/figures/discount-partitions.png')

if __name__ == '__main__':
    discount_partitions()
