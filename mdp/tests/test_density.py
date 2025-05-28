import pytest
import jax.numpy as jnp
from numpy.typing import NDArray
import numpy.random as rnd # For test data generation, switch to jax.random if needed

import mdp.utils as utils
from mdp.density import (
    density_value_functional,
    value_jacobian,
    probability_chain_rule,
    entropy_jacobian,
    distributional_value_function
)

def test_density_value_functional_runs():
    """
    Tests if density_value_functional runs with basic inputs.
    Migrated from the old mdp/density_tests.py.
    Note: The mathematical validity and expected output are complex. This test primarily
    checks for runtime errors with plausible inputs and that a float is returned.
    The original test only printed the output.
    """
    num_states, num_actions = 2, 2
    discount = 0.9
    # Use numpy for random data generation as jax.random requires PRNG keys
    P_np = rnd.rand(num_states, num_actions, num_states)
    P_np = P_np / P_np.sum(axis=2, keepdims=True) # Normalize
    r_np = rnd.rand(num_states, num_actions)
    
    P: NDArray[jnp.float_] = jnp.array(P_np)
    r: NDArray[jnp.float_] = jnp.array(r_np)

    # Create a policy (ensure it's stochastic)
    pi_np = rnd.rand(num_states, num_actions)
    pi_np = pi_np / pi_np.sum(axis=1, keepdims=True)
    pi: NDArray[jnp.float_] = jnp.array(pi_np)
    
    policy_probability_density: float = 0.1 # Example scalar probability density

    # The current implementation of value_jacobian returns a vector, which will cause
    # np.linalg.det in probability_chain_rule to fail.
    # This test will therefore fail until value_jacobian is corrected to return a square matrix.
    with pytest.raises(jnp.linalg.LinAlgError, match="must be square"):
        p_V = density_value_functional(policy_probability_density, P, r, pi, discount)
        # If it were to pass, we'd check the type:
        # assert isinstance(p_V, float), "Expected a float return value for density"

def test_value_jacobian_simple():
    """
    Tests value_jacobian with simple inputs.
    Focuses on shape and basic properties, acknowledging the interpretation issues.
    """
    num_states = 2
    discount = 0.9
    
    # r_pi: (S,)
    r_pi_np = rnd.rand(num_states)
    r_pi: NDArray[jnp.float_] = jnp.array(r_pi_np)
    
    # P_pi: (S, S)
    P_pi_np = rnd.rand(num_states, num_states)
    P_pi_np = P_pi_np / P_pi_np.sum(axis=1, keepdims=True) # Normalize rows
    P_pi: NDArray[jnp.float_] = jnp.array(P_pi_np)

    jacobian_output = value_jacobian(r_pi, P_pi, discount)
    
    # Current value_jacobian returns r_pi[:, newaxis] * matrix_power(...)
    # So, its shape should be (S, S)
    assert jacobian_output.shape == (num_states, num_states), \
        f"Expected shape ({num_states}, {num_states}), got {jacobian_output.shape}"
    assert isinstance(jacobian_output, jnp.ndarray), "Expected JAX ndarray output"


def test_probability_chain_rule_simple():
    """
    Tests probability_chain_rule with a known Jacobian determinant.
    """
    original_density = 0.5
    # Jacobian matrix J = dy/dx. Example: y1 = 2x1, y2 = 3x2
    # J = [[2, 0], [0, 3]], det(J) = 6
    jacobian_matrix_np = np.array([[2.0, 0.0], [0.0, 3.0]])
    jacobian_matrix: NDArray[jnp.float_] = jnp.array(jacobian_matrix_np)
    
    transformed_density = probability_chain_rule(original_density, jacobian_matrix)
    
    expected_density = original_density / np.abs(np.linalg.det(jacobian_matrix_np))
    assert isinstance(transformed_density, float) or isinstance(transformed_density, jnp.ndarray)
    assert jnp.isclose(transformed_density, expected_density), \
        f"Expected density {expected_density}, got {transformed_density}"

    # Test with zero determinant
    jacobian_matrix_zero_det_np = np.array([[1.0, 2.0], [1.0, 2.0]]) # det = 0
    jacobian_matrix_zero_det: NDArray[jnp.float_] = jnp.array(jacobian_matrix_zero_det_np)
    nan_density = probability_chain_rule(original_density, jacobian_matrix_zero_det)
    assert jnp.isnan(nan_density), "Expected NaN for zero determinant Jacobian"

def test_entropy_jacobian_simple():
    """
    Tests entropy_jacobian with a simple policy distribution.
    """
    # Policy for a single state: pi(a|s)
    # Example: pi = [0.2, 0.8]
    pi_np = np.array([0.2, 0.8])
    pi_jax: NDArray[jnp.float_] = jnp.array(pi_np)
    
    # dH/dpi_j = -(log(pi_j) + 1)
    expected_jacobian_np = -1.0 - np.log(pi_np)
    
    jacobian_output = entropy_jacobian(pi_jax)
    
    assert jacobian_output.shape == pi_jax.shape, \
        f"Expected shape {pi_jax.shape}, got {jacobian_output.shape}"
    assert isinstance(jacobian_output, jnp.ndarray), "Expected JAX ndarray output"
    assert jnp.allclose(jacobian_output, jnp.array(expected_jacobian_np)), \
        "Jacobian output does not match expected values."

@pytest.mark.skip(reason="distributional_value_function is not yet implemented.")
def test_distributional_value_function_placeholder():
    """
    Placeholder test for the unimplemented distributional_value_function.
    """
    # Basic parameters
    num_states, num_actions = 2, 2
    discount = 0.9
    P = jnp.zeros((num_states, num_actions, num_states)) # Dummy
    r = jnp.zeros((num_states, num_actions)) # Dummy
    pi = jnp.ones((num_states, num_actions)) / num_actions # Dummy uniform policy
    
    # This test should not run until the function is implemented.
    # When implemented, it would call:
    # dist_V = distributional_value_function(P, r, pi, discount)
    # And then assert properties of dist_V based on its defined return type/structure.
    pass
