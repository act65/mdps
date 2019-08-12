import jax.numpy as np
import numpy.random as rnd

from entropy import *
import mdp.utils as utils

class TestDensity():
    def __init__(self):
        self.test_density()

    @staticmethod
    def test_density():
        mdp = utils.build_random_mdp(2, 2, 0.9)

        pi = utils.softmax(rnd.standard_normal((2,2)), axis=1)
        p_V = density_value_functional(0.1, mdp.P, mdp.r, pi, 0.9)
        print(p_V)

if __name__ == '__main__':
    TestDensity()
