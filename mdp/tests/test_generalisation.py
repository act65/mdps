import pytest
import jax.numpy as jnp
from numpy.typing import NDArray

import mdp.utils as utils
from mdp.generalisation import adjusted_value_iteration

def test_adjusted_value_iteration_runs():
    """
    Tests if adjusted_value_iteration runs and Q-values change.
    Does not verify correctness of the adjustment logic, only that it executes.
    """
    num_states, num_actions = 2, 2
    discount = 0.9
    learning_rate = 0.1

    mdp = utils.build_random_mdp(num_states, num_actions, discount)
    
    # D_matrix and K_matrix: For simplicity, using identity matrices.
    # Their exact role and shape depend on the specific generalisation theory being applied.
    # If Q is (S,A), and K, D are (S,S), the dot products in `adjusted_value_iteration`
    # will apply K and D to each action's Q-values per state, assuming broadcasting.
    # Or, if error is (S*A, 1), then K, D could be (S*A, S*A).
    # The refactored code assumes K, D are (S,S) and operate on Q (S,A) by effectively
    # applying the transformation per action column: K @ D @ Q_column.
    # More precisely, K @ (D @ Q) where (D@Q)[s,a] = sum_s' D[s,s'] Q[s',a].
    
    D_matrix = jnp.eye(num_states)
    K_matrix = jnp.eye(num_states)

    # Initial Q-values (random)
    Q_initial = rnd.standard_normal((num_states, num_actions))
    Q_initial_jnp = jnp.array(Q_initial)

    update_fn = adjusted_value_iteration(mdp, learning_rate, D_matrix, K_matrix)
    Q_next = update_fn(Q_initial_jnp)

    assert Q_next.shape == Q_initial_jnp.shape, "Output Q-values shape mismatch."
    assert not jnp.allclose(Q_next, Q_initial_jnp), "Q-values did not change after update."
    assert isinstance(Q_next, jnp.ndarray), "Output is not a JAX array."

def test_adjusted_value_iteration_specific_case():
    """
    Tests adjusted_value_iteration with a very simple deterministic MDP
    and specific D, K matrices to trace the calculation for one step.
    """
    num_states, num_actions = 2, 1 # Single action for simplicity
    discount = 0.9
    learning_rate = 1.0 # Make update effect clear

    # MDP: S0 --A0--> S0 (P=1), R=1
    #      S1 --A0--> S1 (P=1), R=2
    P = jnp.array([
        [[1.0], [0.0]], # P(S0|S0,A0)=1, P(S0|S1,A0)=0
        [[0.0], [1.0]]  # P(S1|S0,A0)=0, P(S1|S1,A0)=1
    ]) # P[s_next, s_current, action]
    r = jnp.array([[1.0], [2.0]]) # r[s_current, action]
    d0 = jnp.array([[1.0], [0.0]])
    mdp = utils.MDP(S=num_states, A=num_actions, P=P, r=r, discount=discount, d0=d0)

    # Initial Q-values
    Q_initial = jnp.array([[0.0], [0.0]]) # Q[s,a]

    # D = I, K = I (standard Bellman update if K@D@Error = Error)
    D_identity = jnp.eye(num_states)
    K_identity = jnp.eye(num_states)

    update_fn_identity = adjusted_value_iteration(mdp, learning_rate, D_identity, K_identity)
    Q_next_identity = update_fn_identity(Q_initial)
    
    # T_bellman(Q_initial):
    # Q(S0,A0) = r(S0,A0) + g * max_a' Q(S0,a')_from_V_max_of_Q_initial -> V_max(Q_initial) = [0,0]
    #          = 1.0 + 0.9 * (P(S0|S0,A0)*V_max[0] + P(S1|S0,A0)*V_max[1])
    #          = 1.0 + 0.9 * (1.0*0 + 0.0*0) = 1.0
    # Q(S1,A0) = r(S1,A0) + g * (P(S0|S1,A0)*V_max[0] + P(S1|S1,A0)*V_max[1])
    #          = 2.0 + 0.9 * (0.0*0 + 1.0*0) = 2.0
    # T_Q_initial = [[1.0], [2.0]]
    # TD_error = T_Q_initial - Q_initial = [[1.0], [2.0]]
    # Adjustment = I @ I @ TD_error = TD_error
    # Q_next = Q_initial + lr * TD_error = [[0],[0]] + 1.0 * [[1],[2]] = [[1],[2]]
    expected_Q_next_identity = jnp.array([[1.0], [2.0]])
    np.testing.assert_allclose(Q_next_identity, expected_Q_next_identity, atol=1e-6)

    # Test with non-identity K (D is still identity)
    # K swaps the TD errors of the states
    K_swap = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    update_fn_k_swap = adjusted_value_iteration(mdp, learning_rate, D_identity, K_swap)
    Q_next_k_swap = update_fn_k_swap(Q_initial)
    # TD_error = [[1.0], [2.0]]
    # Adjustment = K_swap @ TD_error = [[0,1],[1,0]] @ [[1],[2]] = [[2],[1]] (if matmul applies row-wise to error vectors)
    # This interpretation of K @ D @ TD_error as (K@D) @ TD_error_matrix is what the code does.
    # Q_next = Q_initial + lr * Adjustment = [[0],[0]] + 1.0 * [[2],[1]] = [[2],[1]]
    expected_Q_next_k_swap = jnp.array([[2.0], [1.0]])
    np.testing.assert_allclose(Q_next_k_swap, expected_Q_next_k_swap, atol=1e-6)

    # Test with non-identity D (K is identity)
    D_scale_s0 = jnp.array([[0.5, 0.0], [0.0, 1.0]]) # Scale TD error of S0 by 0.5
    update_fn_d_scale = adjusted_value_iteration(mdp, learning_rate, D_scale_s0, K_identity)
    Q_next_d_scale = update_fn_d_scale(Q_initial)
    # TD_error = [[1.0], [2.0]]
    # D @ TD_error = [[0.5,0],[0,1]] @ [[1.0],[2.0]] (element-wise if applied per action, or matrix product)
    # If (D @ TD_error_matrix) where TD_error_matrix is (S,A)
    # D_TD_error = [[0.5*1.0],[1.0*2.0]] = [[0.5],[2.0]]
    # Adjustment = I @ D_TD_error = D_TD_error
    # Q_next = Q_initial + lr * Adjustment = [[0],[0]] + 1.0 * [[0.5],[2.0]] = [[0.5],[2.0]]
    expected_Q_next_d_scale = jnp.array([[0.5], [2.0]])
    np.testing.assert_allclose(Q_next_d_scale, expected_Q_next_d_scale, atol=1e-6)

# TODO: Add tests for the commented-out `corrected_value_iteration` if it gets implemented.
