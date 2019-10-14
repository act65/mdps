import jax.numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import copy
import functools

from mdp.utils import *
from mdp.search_spaces import *

def clipped_stack(x, n=1000):
    m = len(x)
    k = m//n if m//n > 0 else 1
    return [x[i] for i in range(0, m, k)]

def generate_vi_sgd_vs_mom(mdp, init, lr=0.01):
    print('\nRunning VI SGD vs Mom')

    # sgd
    qs = utils.solve(value_iteration(mdp, lr), init)
    qs = clipped_stack(qs,1000)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='m', label='gd')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='m', marker='x')

    # momentum
    init = (init, np.zeros_like(init))
    qs = utils.solve(momentum_bundler(value_iteration(mdp, lr), 0.99), init)
    qs = clipped_stack(qs,1000)
    vs = np.vstack([np.max(q[0], axis=1) for q in qs])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='momentum')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='autumn', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    plt.title('SGD: {}, Mom {}, Lr: {}'.format(n, m, lr))
    plt.legend()

    plt.savefig('figs/vi_sgd-vs-vi_mom_{}.png'.format(lr))
    plt.close()


def generate_pvi_vs_vi(mdp, init):
    print('\nRunning PVI vs VI')
    lr = 0.01

    # pvi
    core_init = random_parameterised_matrix(2, 2, 32, 4)
    core_init = approximate(init, core_init)
    params = utils.solve(parameterised_value_iteration(mdp, lr/len(core_init)), core_init)
    vs = np.vstack([np.max(build(c), axis=1) for c in params])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='pvi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='autumn', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    # vi
    qs = utils.solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='vi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    plt.title('VI: {}, PVI {}'.format(n, m))
    plt.legend()
    # plt.colorbar()

    plt.savefig('figs/vi-vs-pvi.png', dpi=300)
    plt.close()

def generate_cvi(mdp, init):
    print('\nRunning PVI vs VI')
    lr = 0.01

    # vi
    init = np.complex(init, np.zeros_like(init))
    qs = utils.solve(complex_value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='vi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    plt.title('VI: {}, PVI {}'.format(n, m))
    plt.legend()
    # plt.colorbar()

    plt.savefig('figs/vi-vs-pvi.png', dpi=300)
    plt.close()


def generate_PG_vs_VI(mdp, init):
    print('\nRunning PG vs VI')
    lr = 0.001

    # PG
    logits = utils.solve(policy_gradient_iteration_logits(mdp, lr), init)
    vs = np.vstack([value_functional(mdp.P, mdp.r, softmax(logit), mdp.discount).T for logit in logits])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='PG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    # VI
    v = value_functional(mdp.P, mdp.r, softmax(init), mdp.discount)
    init = np.einsum('ijk,jl->jk', mdp.P, v)  # V->Q
    qs = utils.solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='VI')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='autumn', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')
    plt.legend()
    plt.title('PG: {}, VI {}'.format(n, m))
    # plt.colorbar()

    plt.savefig('figs/pg-vs-vi.png')
    plt.close()

def generate_PG_vs_PPG(mdp, init):
    print('\nRunning PG vs PPG')
    lr = 0.1

    # PPG
    core_init = random_parameterised_matrix(2, 2, 32, 8)
    core_init = approximate(init, core_init)
    all_params = utils.solve(parameterised_policy_gradient_iteration(mdp, lr/len(core_init)), core_init)
    vs = np.vstack([np.max(value_functional(mdp.P, mdp.r, softmax(build(params), axis=-1), mdp.discount), axis=1)
                    for params in all_params])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='PPG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    # PG
    logits = utils.solve(policy_gradient_iteration_logits(mdp, lr), init)
    vs = np.vstack([np.max(value_functional(mdp.P, mdp.r, softmax(logit, axis=-1), mdp.discount), axis=1) for logit in logits])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='PG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='autumn', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    plt.title('PG: {}, PPG {}'.format(n, m))
    plt.legend()
    # plt.colorbar()

    plt.savefig('figs/pg-vs-ppg.png', dpi=300)
    plt.close()

def generate_mppg_vs_mpg(mdp, init):
    print('\nRunning MPG vs MPPG')
    lr = 0.001

    # MPPG
    core_init = random_parameterised_matrix(2, 2, 32, 8)
    core_init = approximate(init, core_init)
    core_init = (core_init, [np.zeros_like(c) for c in core_init])
    all_params = utils.solve(momentum_bundler(parameterised_policy_gradient_iteration(mdp, lr), 0.9), core_init)
    vs = np.vstack([np.max(value_functional(mdp.P, mdp.r, softmax(build(params), axis=-1), mdp.discount), axis=1)
                    for params, mom in all_params])
    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='MPPG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')

    # MPG
    init = (init, np.zeros_like(init))
    logits = utils.solve(momentum_bundler(policy_gradient_iteration_logits(mdp, lr), 0.9), init)
    vs = np.vstack([np.max(value_functional(mdp.P, mdp.r, softmax(logit, axis=-1), mdp.discount), axis=1) for logit, mom in logits])
    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='MPG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='autumn', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    plt.title('PG: {}, PPG {}'.format(n, m))
    plt.legend()
    # plt.colorbar()

    plt.savefig('figs/mpg-vs-mppg.png', dpi=300)
    plt.close()

def generate_mpvi_vs_mvi(mdp, init):
    print('\nRunning MPVI vs MVI')
    lr = 1e-2
    # mpvi
    core_init = random_parameterised_matrix(2, 2, 32, 8)
    core_init = approximate(init, core_init)
    c_init = (core_init, [np.zeros_like(c) for c in core_init])
    params = utils.solve(momentum_bundler(parameterised_value_iteration(mdp, lr), 0.9), c_init)
    vs = np.vstack([np.max(build(c[0]), axis=-1) for c in params])
    m = vs.shape[0]

    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='mpvi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(m-2), cmap='autumn', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='r', marker='x')

    # mvi
    init = (init, np.zeros_like(init))
    qs = utils.solve(momentum_bundler(value_iteration(mdp, lr), 0.9), init)
    vs = np.vstack([np.max(q[0], axis=-1) for q in qs])
    n = vs.shape[0]

    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='mvi')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')


    plt.title('MVI: {}, MPVI {}'.format(n, m))
    plt.legend()

    plt.savefig('figs/mpvi-vs-pvi.png')
    plt.close()




# def generate_mpvi_inits(mdp):
#     print('Running MPVI inits')
#
#     core_init = random_parameterised_matrix(2, 2, 16, 6)
#     init = (core_init, [np.zeros_like(c) for c in core_init])
#     params = utils.solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), init)
#     vs = np.vstack([np.max(build(c[0]), axis=-1) for c in params])
#     m = vs.shape[0]
#     plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='mpvi')
#
#
#     new_init = random_reparameterisation(core_init, 2)
#     # sanity check
#     assert np.isclose(build(core_init), build(new_init), atol=1e-4).all()
#     init = (new_init, [np.zeros_like(c) for c in core_init])
#     params = utils.solve(momentum_bundler(parameterised_value_iteration(mdp, 0.01), 0.9), init)
#     vs = np.vstack([np.max(build(c[0]), axis=-1) for c in params])
#     m = vs.shape[0]
#     plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='spring', label='mpvi')
#
#     plt.savefig('figs/mpvi-inits.png')
#     plt.close()
#     """
#     Hmm. I thought this would change the dynamics.
#     It is because the value is only a linear function of the parameters???
#     """

# def generate_pvi_vs_apvi():
#     print('Running PVI vs APVI')
#     n_states, n_actions = 2, 2
#     lr = 0.01
#
#     mdp = build_random_mdp(n_states, n_actions, 0.9)
#
#     core_init = random_parameterised_matrix(2, 2, 32, 2)
#     # pvi
#     params = utils.solve(parameterised_value_iteration(mdp, lr), core_init)
#     vs = np.hstack([build(c) for c in params]).T
#     m = vs.shape[0]
#     plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', label='pvi')
#
#     # TODO want to visualise.
#     # @jit
#     def K(dQ):
#         return np.tensordot(dQ, dQ, axes=([-1,-2],[-1,-2]))
#
#     dVdw = jit(jacrev(value))
#     dQs = [dVdw(cores) for cores in params][0:10]
#
#     Ks = [sum([K(dq) for dq in dQ]).reshape((mdp.S * mdp.A, mdp.S * mdp.A)) for dQ in dQs]
#
#     n = 100
#     x = np.stack(np.meshgrid(np.linspace(-1,1,n), np.linspace(-1,1,n)), axis=0).reshape((2, n**2))
#     print(x.shape)
#
#     # plt.show()


# pg vs pi

# def argumentparser():
#     parser = argparse.ArgumentParser(description='Visualise losses and returns')
#     parser.add_argument('--logdir', type=str, default='logs',
#                         help='location to save logs')
#     return parser.parse_args()

# def plot(vs1, vs2, save_path):




def generate_model_iteration(mdp, init):
    init = rnd.standard_normal((mdp.S, mdp.S, mdp.A))  # needs its own init. alternatively could find init that matches value of other inits?!?
    pi_star = utils.solve(policy_iteration(mdp), utils.softmax(rnd.standard_normal((mdp.S,mdp.A))))[-1]
    update_fn = model_iteration(mdp, 0.01)
    logits = utils.solve(update_fn, init)

    vs = np.vstack([utils.value_functional(utils.softmax(logit), mdp.r, pi_star, mdp.discount).T for logit in logits])

    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='PG')
    plt.scatter(vs[1:-1, 0], vs[1:-1, 1], c=range(n-2), cmap='spring', s=10)
    plt.scatter(vs[-1, 0], vs[-1, 1], c='g', marker='x')
    # plt.show()




if __name__ == '__main__':
    print('start')
    n_states, n_actions = 2, 2
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    print('\nBuilding polytope')
    pis = utils.gen_grid_policies(11)
    vs = utils.polytope(mdp.P, mdp.r, mdp.discount, pis)
    init = rnd.standard_normal((mdp.S, mdp.A))

    experiments = [
        # functools.partial(generate_vi_sgd_vs_mom, lr=a) for a in np.logspace(-5, 0, 6)
        #
        # generate_vi_sgd_vs_mom,
        # generate_pvi_vs_vi,
        #
        # generate_avi_vs_vi,
        #
        # generate_PG_vs_VI,
        generate_PG_vs_PPG,
        #
        # generate_mpvi_vs_mvi,
        # generate_mppg_vs_mpg,
        #
        # generate_mpvi_inits,
        generate_model_iteration
    ]

    print('\nRunning experiments')
    for exp in experiments:
        plt.figure(figsize=(16,16))
        plt.scatter(vs[:, 0], vs[:, 1], s=10, alpha=0.75)
        exp(mdp, init)
        plt.show()
