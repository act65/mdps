import numpy as np
import src.utils as utils
from src.search_spaces import *

def clip_solver_traj(traj):
    if np.isclose(traj[-1], traj[-2], 1e-8).all():
        return traj[:-1]
    else:
        return traj

mdp = utils.build_random_mdp(2, 2, 0.5)
init = utils.softmax(rnd.standard_normal((mdp.S, mdp.A)), axis=1)
pi_traj = clip_solver_traj(utils.solve(policy_iteration(mdp), init))
print(pi_traj)
