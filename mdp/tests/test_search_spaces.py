import numpy as np
import numpy.random as rnd
import mdp.utils as utils
from mdp.search_spaces import policy_iteration

def clip_solver_traj(traj):
    if np.isclose(traj[-1], traj[-2], 1e-8).all():
        return traj[:-1]
    else:
        return traj

def test_search_spaces_simple():
    mdp = utils.build_random_mdp(2, 2, 0.5)
    init = utils.softmax(rnd.standard_normal((mdp.S, mdp.A)), axis=1)
    pi_traj = clip_solver_traj(utils.solve(policy_iteration(mdp), init))
    print(pi_traj)
    assert len(pi_traj) > 0

