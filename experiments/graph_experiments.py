import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

import mdp.graph as graph
import mdp.utils as utils
import mdp.search_spaces as search_spaces

def graph_PI():
    n_states = 10
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    print('n pis: {}'.format(len(det_pis)))
    mdp = utils.build_random_sparse_mdp(n_states, n_actions, 0.5)

    A = graph.mdp_topology(det_pis)
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, iterations=200)

    basis = graph.construct_mdp_basis(det_pis, mdp)

    init_pi = utils.softmax(np.random.standard_normal((n_states, n_actions)))
    init_v = utils.value_functional(mdp.P, mdp.r, init_pi, mdp.discount).squeeze()
    a = graph.sparse_coeffs(basis, init_v, lr=0.1)

    pis = utils.solve(search_spaces.policy_iteration(mdp), init_pi)
    print("\n{} policies to vis".format(len(pis)))

    for i, pi in enumerate(pis[:-1]):
        print('Iteration: {}'.format(i))
        v = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount).squeeze()
        a = graph.sparse_coeffs(basis, v, lr=0.1, a_init=a)
        plt.figure(figsize=(16,16))
        nx.draw(G, pos, node_color=a, node_size=150)
        # plt.show()
        plt.savefig('figs/pi_graphs/{}.png'.format(i))
        plt.close()

def value_graph():

    # vs = [np.sum(utils.value_functional(mdp.P, mdp.r, pi, mdp.discount).squeeze()**2) for pi in det_pis]
    # plt.figure(figsize=(16,16))
    # nx.draw(G, pos, node_color=vs, node_size=150)
    # plt.savefig('figs/pi_graphs/val.png')
    # plt.close()

    n_states = 10
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    n = len(det_pis)
    print('n pis: {}'.format(n))
    # how does discount effect these!?
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    values = [utils.value_functional(mdp.P, mdp.r, pi, mdp.discount).squeeze() for pi in det_pis]
    vs = np.stack(values).reshape((n, n_states))
    W = 1/(np.abs(np.sum(vs[None, :, :] - vs[:, None, :], axis=-1)) + 1e-8)
    A = graph.mdp_topology(det_pis)
    adj = A*W
    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, iterations=200)

    plt.figure(figsize=(16,16))
    nx.draw(G, pos, node_color=[np.sum(v) for v in values], node_size=150)
    plt.savefig('figs/value_graphs/value_graph-{}-{}.png'.format(n_states, n_actions))
    plt.close()

def graph_laplacian_spectra(A):
    D = np.sum(A, axis=0)
    L = np.diag(D) - A
    u, v = np.linalg.eig(L)
    return u, v

def value_graph_laplacian():
    n_states = 8
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    n = len(det_pis)
    print('n pis: {}'.format(n))
    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    values = [utils.value_functional(mdp.P, mdp.r, pi, mdp.discount).squeeze() for pi in det_pis]
    Vs = np.stack(values).reshape((n, n_states))
    A = graph.mdp_topology(det_pis)

    W = 1/(np.abs(np.sum(Vs[None, :, :] - Vs[:, None, :], axis=-1)) + 1e-8)
    adj = A*W

    G = nx.from_numpy_array(adj)
    pos = nx.spring_layout(G, iterations=200)
    plt.figure(figsize=(16,16))
    nx.draw(G, pos, node_color=[np.sum(v) for v in values], node_size=150)
    plt.savefig('figs/value_graphs/value_graph-{}-{}.png'.format(n_states, n_actions))
    plt.close()

    # how can you calulate expected eignenvalues!?
    # observation. the underlying complexity of the value topology is linear!?!?
    # how hard is it to estimate the main eigen vec from noisy observations!?
    # that would tell us the complexity!?!?
    for i, alpha in enumerate(np.linspace(0, 1, 10)):
        us = []
        for _ in range(50):
            vs = Vs + alpha*np.random.standard_normal(Vs.shape)
            W = 1/(np.abs(np.sum(vs[None, :, :] - vs[:, None, :], axis=-1)) + 1e-8)
            adj = A*W

            u, v = graph_laplacian_spectra(adj)
            us.append(u)
        us = np.stack(us, axis=0)
        mean = np.mean(us, axis=0)
        var = np.var(us, axis=0)
        plt.bar(range(len(mean)), mean, yerr=np.sqrt(var))
        plt.savefig('figs/value_graphs/{}-lap.png'.format(i))
        plt.close()

def value_graph_laplacians():
    n_states = 8
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    N = len(det_pis)
    print('n pis: {}'.format(N))
    for i in range(1):
        mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

        values = [utils.value_functional(mdp.P, mdp.r, pi, mdp.discount).squeeze() for pi in det_pis]
        Vs = np.stack(values).reshape((N, n_states))
        A = graph.mdp_topology(det_pis)

        W = np.exp(-np.linalg.norm(Vs[None, :, :] - Vs[:, None, :], ord=np.inf, axis=-1)+1e-8)

        # mVs = np.mean(Vs, axis=0)  # n_states
        # W = np.dot((Vs - mVs) , (Vs - mVs).T)
        adj = W * A

        G = nx.from_numpy_array(adj)
        pos = nx.spectral_layout(G) #, iterations=500)
        plt.figure(figsize=(16,16))
        nx.draw(G, pos, node_color=[np.sum(v) for v in values], node_size=150)
        plt.savefig('figs/value_graphs/{}-value_graph-{}-{}.png'.format(i, n_states, n_actions))
        plt.close()

        u, v = graph_laplacian_spectra(adj)
        plt.figure(figsize=(8,8))
        plt.bar(range(len(u)), u)
        plt.savefig('figs/value_graphs/{}-lap.png'.format(i))
        plt.close()

        plt.figure(figsize=(16,16))
        n = 5
        for j in range(n*n):
            plt.subplot(n,n,j+1)
            nx.draw(G, pos, node_color=u[10*j] * v[10*j], node_size=150)
        plt.savefig('figs/value_graphs/{}-spectra.png'.format(i, n_states, n_actions))
        plt.close()

def graph_PG():
    # ffmpeg -framerate 10 -start_number 0 -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    n_states = 6
    n_actions = 4

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    print('n pis: {}'.format(len(det_pis)))
    mdp = utils.build_random_mdp(n_states, n_actions, 0.9)

    A = graph.mdp_topology(det_pis)
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, iterations=200)

    basis = graph.construct_mdp_basis(det_pis, mdp)

    init_logits = np.random.standard_normal((n_states, n_actions))
    init_v = utils.value_functional(mdp.P, mdp.r, utils.softmax(init_logits), mdp.discount).squeeze()
    a = graph.sparse_coeffs(basis, init_v, lr=0.1)

    print('\nSolving PG')
    pis = utils.solve(search_spaces.policy_gradient_iteration_logits(mdp, 0.1), init_logits)
    print("\n{} policies to vis".format(len(pis)))
    n = len(pis)
    # pis = pis[::n//100]
    pis = pis[0:20]

    for i, pi in enumerate(pis[:-1]):
        print('Iteration: {}'.format(i))
        v = utils.value_functional(mdp.P, mdp.r, pi, mdp.discount).squeeze()
        a = graph.sparse_coeffs(basis, v, lr=0.1, a_init=a)
        plt.figure()
        nx.draw(G, pos, node_color=a)
        # plt.show()
        plt.savefig('figs/pg_graphs/{}.png'.format(i))
        plt.close()


if __name__ == '__main__':
    # value_graph()
    # graph_PI()
    # graph_PG()
    # value_graph_laplacian()
    value_graph_laplacians()
