import os
import json, codecs
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mdp.utils as utils
import mdp.search_spaces as ss


if __name__ == '__main__':
    fname = 'test1.json'
    experiments = []
    with open(fname, 'r') as f:
        for line in f:

            exp = json.loads(line)
            exp = {list(exp.keys())[0]: [np.asarray(v) for v in list(exp.values())[0]]}
            experiments.append(exp)
    # print(experiments)

    for exp in experiments:
        for k, v in exp.items():
            print(k)
            print([x.shape for x in v])
