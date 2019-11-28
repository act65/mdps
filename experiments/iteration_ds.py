import os
import json, codecs
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mdp.utils as utils
import mdp.search_spaces as ss

def value_iteration(mdp, pis, lr):
    trajs = []

    for pi in pis:
        init_V = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount)

        traj = utils.solve(ss.value_iteration(mdp, lr), init_V)
        v_star = traj[-1]
        trajs.append(traj)
    return trajs

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ =='__main__':
    rnd.seed(42)
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)
    pis = utils.gen_grid_policies(4)

    use_momentum = False
    fname = 'test1.json'
    with open(fname, 'w') as f:

        for lr in np.logspace(-2, -1, 2):
            traj = value_iteration(mdp, pis, lr)

            data = {'{}-{}-{}'.format(value_iteration.__name__, lr, use_momentum) : [np.array(t).tolist() for t in traj]}
            s = json.dumps(data, cls=NumpyEncoder)
            f.write(s+'\n')
