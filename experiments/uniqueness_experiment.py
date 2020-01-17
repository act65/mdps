import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mdp.utils as utils
import mdp.search_spaces as ss

if __name__ == '__main__':
    P = np.zeros((2,2,2))
    a = np.random.random((2,2))
    a = a/np.sum(a,axis=0)
    P[:,:,0] = a
    P[:,:,1] = a
    # print(P[:, :, 0])
    # print(P[:, :, 1])

    r = np.array([
        [1, 0],
        [0, 1]
    ])
    discount = 0.5

    pis = utils.gen_grid_policies(21)
    vs = utils.polytope(P, r, discount, pis)
    plt.scatter(vs[:, 0], vs[:, 1])
    plt.show()
