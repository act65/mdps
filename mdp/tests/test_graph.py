import pytest
import jax.numpy as jnp
from numpy.typing import NDArray
import numpy.random as rnd # For test data generation
import numpy as np # For assertions and some test data setup

import mdp.utils as utils
from mdp.graph import (
    construct_mdp_basis,
    mdp_topology,
    estimate_coeffs,
    mse,
    sparse_coeffs
)

# --- Helper function to create deterministic policies for tests ---
def _create_deterministic_policies(num_states: int, num_actions: int, policy_indices: List[List[int]]) -> List[NDArray[jnp.float_]]:
    """
    Creates a list of one-hot encoded deterministic policies.
    policy_indices is a list of lists, where each inner list contains the action for each state.
    e.g., [[0,1], [1,0]] for 2 states, 2 actions means:
    - policy 0: state 0 -> action 0, state 1 -> action 1
    - policy 1: state 0 -> action 1, state 1 -> action 0
    """
    policies = []
    for indices in policy_indices:
        policy = jnp.zeros((num_states, num_actions), dtype=jnp.float32)
        for state_idx, action_idx in enumerate(indices):
            policy = policy.at[state_idx, action_idx].set(1.0)
        policies.append(policy)
    return policies

# --- Tests for mdp/graph.py functions ---

def test_construct_mdp_basis_simple():
    """Tests construct_mdp_basis with a small MDP and known deterministic policies."""
    num_states, num_actions = 2, 2
    discount = 0.9
    
    # Simple MDP: P[s,a,s'] = 1 if s'=a for s=0, else s'=s for s=1. r[s,a] = a.
    P = jnp.zeros((num_states, num_actions, num_states))
    P = P.at[0, 0, 0].set(1.0) # S0, A0 -> S0
    P = P.at[0, 1, 1].set(1.0) # S0, A1 -> S1
    P = P.at[1, 0, 0].set(1.0) # S1, A0 -> S0
    P = P.at[1, 1, 1].set(1.0) # S1, A1 -> S1
    
    r = jnp.array([[0., 1.], [0., 1.]]) # r(s,a) = a
    
    mdp = utils.MDP(S=num_states, A=num_actions, P=P, r=r, discount=discount, d0=None)

    # Deterministic policies:
    # Pol 0: S0->A0, S1->A0
    # Pol 1: S0->A1, S1->A1
    det_policies_indices = [[0,0], [1,1]]
    det_policies = _create_deterministic_policies(num_states, num_actions, det_policies_indices)

    basis_matrix = construct_mdp_basis(det_policies, mdp)

    assert basis_matrix.shape == (num_states, len(det_policies)), \
        f"Expected basis shape ({num_states}, {len(det_policies)}), got {basis_matrix.shape}"

    # Manual calculation for V_pol0 (S0->A0, S1->A0):
    # V0(S0) = r(S0,A0) + g * V0(S0) = 0 + 0.9 * V0(S0) => V0(S0) = 0
    # V0(S1) = r(S1,A0) + g * V0(S0) = 0 + 0.9 * 0    => V0(S1) = 0
    # V_pol0 should be [0, 0]
    
    # Manual calculation for V_pol1 (S0->A1, S1->A1):
    # V1(S0) = r(S0,A1) + g * V1(S1) = 1 + 0.9 * V1(S1)
    # V1(S1) = r(S1,A1) + g * V1(S1) = 1 + 0.9 * V1(S1) => 0.1 * V1(S1) = 1 => V1(S1) = 10
    # V1(S0) = 1 + 0.9 * 10 = 10
    # V_pol1 should be [10, 10]
    
    expected_basis_col0 = jnp.array([0., 0.])
    expected_basis_col1 = jnp.array([10., 10.])
    
    np.testing.assert_allclose(basis_matrix[:, 0], expected_basis_col0, atol=1e-5)
    np.testing.assert_allclose(basis_matrix[:, 1], expected_basis_col1, atol=1e-5)


def test_mdp_topology_simple():
    """Tests mdp_topology with a few deterministic policies."""
    num_states, num_actions = 2, 2
    # Policies:
    # P0: S0->A0, S1->A0
    # P1: S0->A0, S1->A1 (differs from P0 in S1)
    # P2: S0->A1, S1->A0 (differs from P0 in S0)
    # P3: S0->A1, S1->A1 (differs from P0 in S0, S1)
    det_policies_indices = [[0,0], [0,1], [1,0], [1,1]]
    det_policies = _create_deterministic_policies(num_states, num_actions, det_policies_indices)

    adj_matrix = mdp_topology(det_policies)

    assert adj_matrix.shape == (len(det_policies), len(det_policies)), "Adjacency matrix shape mismatch."

    # Expected adjacency:
    #    P0 P1 P2 P3
    # P0 0  1  1  0  (P0 differs from P1 in 1 state, P2 in 1, P3 in 2)
    # P1 1  0  0  1  (P1 differs from P0 in 1, P2 in 2, P3 in 1)
    # P2 1  0  0  1  (P2 differs from P0 in 1, P1 in 2, P3 in 1)
    # P3 0  1  1  0  (P3 differs from P0 in 2, P1 in 1, P2 in 1)
    expected_adj_matrix = jnp.array([
        [0., 1., 1., 0.],
        [1., 0., 0., 1.],
        [1., 0., 0., 1.],
        [0., 1., 1., 0.]
    ], dtype=jnp.float32)
    
    np.testing.assert_array_equal(adj_matrix, expected_adj_matrix)

def test_mdp_topology_empty():
    """Tests mdp_topology with an empty list of policies."""
    adj_matrix = mdp_topology([])
    assert adj_matrix.shape == (0,0), "Expected (0,0) shape for empty input."


def test_estimate_coeffs_exact():
    """Tests estimate_coeffs when target_vector is an exact combination of basis vectors."""
    num_states = 3
    num_basis_vectors = 2
    
    # Basis: columns are basis vectors
    basis_np = np.array([[1., 0.], [0., 1.], [1., 1.]]) # (S, N_basis)
    basis_jnp = jnp.array(basis_np)
    
    # True coefficients
    true_coeffs_np = np.array([2.0, 3.0]) # (N_basis,)
    
    # Target vector = basis @ true_coeffs
    target_vector_np = basis_np @ true_coeffs_np # (S,)
    target_vector_jnp = jnp.array(target_vector_np)

    estimated_coeffs = estimate_coeffs(basis_jnp, target_vector_jnp)

    assert estimated_coeffs.shape == (num_basis_vectors,), "Coefficients shape mismatch."
    np.testing.assert_allclose(estimated_coeffs, true_coeffs_np, atol=1e-6)

def test_mse_simple():
    """Tests mse with simple vectors."""
    vec_x_jnp = jnp.array([1., 2., 3.])
    vec_y_jnp = jnp.array([1., 1., 4.])
    # Differences: [0, 1, -1]
    # Squared differences: [0, 1, 1]
    # Sum of squared differences: 2
    expected_mse = 2.0
    
    calculated_mse = mse(vec_x_jnp, vec_y_jnp)
    assert isinstance(calculated_mse, jnp.ndarray) # Should be a JAX scalar array
    assert calculated_mse.ndim == 0 # Scalar
    assert jnp.isclose(calculated_mse, expected_mse)

    # Test with zero vectors
    vec_z = jnp.zeros(3)
    assert jnp.isclose(mse(vec_z, vec_z), 0.0)


def test_sparse_coeffs_runs():
    """
    Tests if sparse_coeffs runs and produces output of the correct shape.
    Verifying exact coefficient values is hard without a trivial setup or known solution.
    """
    num_states = 5
    num_basis_vectors = 10 # Number of deterministic policies
    
    # Create a dummy MDP and basis (from test_sparse_estimation in old file)
    # mdp = utils.build_random_mdp(num_states, 2, 0.9) # Assuming 2 actions
    # det_policies_indices = [[i % 2 for _ in range(num_states)] for i in range(num_basis_vectors)] # Dummy policies
    # det_policies = _create_deterministic_policies(num_states, 2, det_policies_indices)
    # basis_jnp = construct_mdp_basis(det_policies, mdp)

    # Simpler basis and target for testing execution
    basis_jnp = jnp.array(rnd.rand(num_states, num_basis_vectors))
    target_vector_jnp = jnp.array(rnd.rand(num_states))

    initial_coeffs_params_jnp = jnp.array(rnd.rand(num_basis_vectors))

    # Test with very few iterations for speed
    coeffs_params_output = sparse_coeffs(
        basis_jnp, 
        target_vector_jnp, 
        initial_coefficients_params=initial_coeffs_params_jnp,
        max_iterations=5, # Keep low for speed
        learning_rate=1e-3 
    )

    assert coeffs_params_output.shape == (num_basis_vectors,), \
        f"Expected output shape ({num_basis_vectors},), got {coeffs_params_output.shape}"
    assert isinstance(coeffs_params_output, jnp.ndarray), "Expected JAX ndarray output."

    # Optional: Check if loss decreased (requires access to loss or initial/final loss)
    # This would involve calling sparse_loss directly.
    # def get_loss(params_x, basis, target, reg_strength):
    #     alpha = utils.softmax(params_x)
    #     return mse(jnp.dot(basis, alpha), target) + reg_strength * utils.entropy(alpha)

    # initial_loss = get_loss(initial_coeffs_params_jnp, basis_jnp, target_vector_jnp, 1e-6)
    # final_loss = get_loss(coeffs_params_output, basis_jnp, target_vector_jnp, 1e-6)
    # assert final_loss < initial_loss, "Loss did not decrease during optimization."
    # Note: For very few iterations, loss decrease isn't guaranteed with momentum / learning rates.
    # This assertion is commented out as it might be flaky.

# It's good practice to also test edge cases, e.g.,
# - mdp_topology with no policies or one policy. (Added empty case for mdp_topology)
# - estimate_coeffs with non-invertible (or poorly conditioned) basis. (pinv handles this)
# - sparse_coeffs with zero regularization.

# The old test_everything and other plotting tests are not directly convertible to unit tests
# as they rely on visual inspection. Their setup logic has been partially used.
# test_estimator from old file seemed like a variant of estimate_coeffs, covered by test_estimate_coeffs_exact.
# test_estimation also covered by test_estimate_coeffs_exact using a more direct setup.
# test_sparse_estimation logic used in test_sparse_coeffs_runs.
# test_topology logic used in test_mdp_topology_simple.
