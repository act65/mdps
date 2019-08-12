import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

import mdp.graph as graph
import mdp.utils as utils
import mdp.search_spaces as search_spaces

def graph_PI():
    n_states = 6
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    mdp = utils.build_random_mdp(n_states, n_actions, 0.9)

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
        plt.figure()
        nx.draw(G, pos, node_color=a)
        # plt.show()
        plt.savefig('figs/pi_graphs/{}.png'.format(i))
        plt.close()

def graph_PG():
    # ffmpeg -framerate 10 -start_number 0 -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    n_states = 6
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    mdp = utils.build_random_mdp(n_states, n_actions, 0.9)

    A = graph.mdp_topology(det_pis)
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, iterations=200)

    basis = graph.construct_mdp_basis(det_pis, mdp)

    init_pi = utils.softmax(np.random.standard_normal((n_states, n_actions)))
    init_v = utils.value_functional(mdp.P, mdp.r, init_pi, mdp.discount).squeeze()
    a = graph.sparse_coeffs(basis, init_v, lr=0.1)

    pis = utils.solve(search_spaces.policy_gradient_iteration_logits(mdp, 0.01), init_pi)
    print("\n{} policies to vis".format(len(pis)))

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
    graph_PG()
