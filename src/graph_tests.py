import numpy as np
import src.graph as graph
import src.utils as utils
import networkx as nx
import matplotlib.pyplot as plt

def test_estimator():
    n_states = 4
    n_actions = 2
    n_det_policies = n_states ** n_actions
    v_det_pis = np.vstack([np.random.random((1,4)) for i in range(n_det_policies)])

    v = np.random.random((4,))

    print(graph.estimate_coeffs(v_det_pis, v))

def test_estimation():
    n_states = 5
    n_actions = 2

    det_pis = utils.get_deterministic_policies(mdp.S, mdp.A)
    mdp = utils.build_random_mdp(n_states, n_actions, 0.9)
    basis = graph.construct_mdp_basis(det_pis, mdp)

    v = np.random.random((n_states, ))
    a = graph.estimate_coeffs(basis.T, v)
    print(a)

def test_sparse_estimation():
    n_states = 5
    n_actions = 2

    mdp = utils.build_random_mdp(n_states, n_actions, 0.9)
    det_pis = utils.get_deterministic_policies(mdp.S, mdp.A)
    basis = graph.construct_mdp_basis(det_pis, mdp)

    v = utils.value_functional(mdp.P, mdp.r, det_pis[2], mdp.discount).squeeze()
    a = graph.sparse_coeffs(basis, v)

    print(a)

def test_topology():
    n_states = 5
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    A = graph.mdp_topology(det_pis)
    print(A)
    G = nx.from_numpy_array(A)
    nx.draw(G)
    plt.show()


def test_everything():
    n_states = 5
    n_actions = 2

    det_pis = utils.get_deterministic_policies(n_states, n_actions)
    mdp = utils.build_random_mdp(n_states, n_actions, 0.9)

    A = graph.mdp_topology(det_pis)
    basis = graph.construct_mdp_basis(det_pis, mdp)

    # v = np.random.random((n_states, ))
    v = utils.value_functional(mdp.P, mdp.r, det_pis[2], mdp.discount).squeeze()
    a = graph.sparse_coeffs(basis, v)

    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, iterations=200)
    nx.draw(G, pos, node_color=a)
    plt.show()

if __name__ == '__main__':
    # test_estimation()
    # test_topology()
    test_everything()
    # test_sparse_estimation()
