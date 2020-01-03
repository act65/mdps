import numpy as np
import numpy.random as rnd
import mdp.utils as utils
import mdp.search_spaces as ss
import matplotlib.pyplot as plt


def generate_vi(mdp, c, lr=0.1):
    init_pi = utils.random_policy(mdp.S,mdp.A)
    init_v = utils.value_functional(mdp.P, mdp.r, init_pi, mdp.discount)
    vs = np.stack(utils.solve(ss.value_iteration(mdp, lr), init_v))[:,:,0]
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c=c, s=30, label='{}'.format(n))
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='viridis', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='m', marker='x')

def generate_pg(mdp, c, lr=0.01):
    init_pi = utils.random_policy(mdp.S,mdp.A)
    init_logit = np.log(init_pi)
    logits = utils.solve(ss.policy_gradient_iteration_logits(mdp, lr), init_logit)
    vs = np.stack([utils.value_functional(mdp.P, mdp.r, utils.softmax(logit), mdp.discount) for logit in logits])[:,:,0]
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c=c, s=30, label='{}'.format(n))
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='viridis', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='m', marker='x')

def generate_pi(mdp, c):
    init_pi = utils.random_policy(mdp.S,mdp.A)
    pis = utils.solve(ss.policy_iteration(mdp), init_pi)
    vs = np.stack([utils.value_functional(mdp.P, mdp.r, pi, mdp.discount) for pi in pis])[:,:,0]
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c=c, s=30, label='{}'.format(n-2))
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='viridis', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='m', marker='x')

    for i in range(len(vs)-2):
        dv = 0.1*(vs[i+1, :] - vs[i, :])
        plt.arrow(vs[i, 0], vs[i, 1], dv[0], dv[1], color=c, alpha=0.5, width=0.005)

if __name__ == '__main__':
    # rnd.seed(42)
    print('start')
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    print('\nBuilding polytope')
    pis = np.stack(utils.gen_grid_policies(41))
    vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)

    plt.figure(figsize=(16,16))
    plt.scatter(vs[:, 0], vs[:, 1], s=10, alpha=0.75)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, c in zip(range(4), colors):
        print('\nRunning experiment {}'.format(i))
        # generate_vi(mdp, c)
        generate_pg(mdp, c)
        # generate_pi(mdp, c)
    plt.legend()
    plt.colorbar()
    plt.show()
