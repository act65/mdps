from typing import List, Tuple, Optional
import jax.numpy as jnp
from jax import grad, jit
from numpy.typing import NDArray
import numpy # For linalg.pinv if jax.numpy.linalg.pinv is not preferred, and for some type hints.
import numpy.random as rnd # For initializations, consider jax.random for pure JAX.

import mdp.utils as utils
import mdp.search_spaces as search_spaces

"""
Related??? https://arxiv.org/pdf/1901.11530.pdf (Found in original code)
This module seems to deal with representing value functions in a basis formed by
value functions of deterministic policies, and analyzing relationships between policies.
"""

def construct_mdp_basis(
    deterministic_policies: List[NDArray[jnp.float_]], 
    mdp: utils.MDP
) -> NDArray[jnp.float_]:
    """
    Constructs a basis for value functions using the value functions of given deterministic policies.

    Each column in the returned basis matrix corresponds to the value function V(s) 
    for one of the provided deterministic policies.

    Args:
        deterministic_policies: A list of deterministic policies. Each policy `pi`
                                is an array of shape (S, A), where S is the number of states
                                and A is the number of actions. pi[s,a] = 1 if action a is
                                taken in state s, and 0 otherwise.
                                More typically, for deterministic policies, it might be (S,)
                                representing the action index for each state.
                                Assuming the input format is (S,A) one-hot encoded.
        mdp: The MDP object (`utils.MDP`) for which the value functions are computed.

    Returns:
        A 2D JAX array of shape (S, num_deterministic_policies), where S is the
        number of states in the MDP. Each column is the value function V(s)
        for a deterministic policy.
    """
    value_functions_list: List[NDArray[jnp.float_]] = [
        utils.value_functional(mdp.P, mdp.r, policy, mdp.discount) 
        for policy in deterministic_policies
    ]
    # Ensure V_det_policy is (S,) before hstack which expects 1D arrays or (S,1)
    # value_functional returns (S,)
    # np.hstack stacks them horizontally, resulting in (S, num_deterministic_policies) if each V is (S,)
    # If V_det_pis are (S,), hstack makes it (S * N_policies,). We want (S, N_policies).
    # Use np.stack and transpose, or np.vstack and transpose if each V is (1,S) or (S,1)
    # If each V_det_policy is (S,), then `jnp.stack(value_functions_list, axis=-1)` is (S, N_policies)
    return jnp.stack(value_functions_list, axis=-1)

def mdp_topology(deterministic_policies: List[NDArray[jnp.float_]]) -> NDArray[jnp.float_]:
    """
    Computes an adjacency matrix representing the topology of deterministic policies.

    Two policies are considered adjacent if they differ in action selection for exactly one state.
    The input policies are assumed to be one-hot encoded (S, A).
    The difference `pi1 - pi2` will have non-zero entries where actions differ.
    Summing `abs(pi1 - pi2)` over actions gives 2 if actions differ for a state (e.g. [1,0]-[0,1] -> abs sums to 2), 0 if same.
    Summing this over states gives `2 * num_differing_states`.
    So, `sum_abs_diff == 2` means policies differ in exactly one state.

    Args:
        deterministic_policies: A list of deterministic policies. Each policy is an
                                array of shape (S, A), one-hot encoded.

    Returns:
        A 2D JAX array (adjacency matrix) of shape (num_policies, num_policies).
        `adjacency_matrix[i, j] = 1.0` if policy `i` and policy `j` differ in exactly
        one state's action, and `0.0` otherwise.
    """
    num_policies: int = len(deterministic_policies)
    if num_policies == 0:
        return jnp.array([]).reshape((0,0)) # Or handle as error
        
    # Stack policies into a single tensor: (num_policies, S, A)
    stacked_policies: NDArray[jnp.float_] = jnp.stack(deterministic_policies)

    # Calculate pairwise differences:
    # policy_i[s,a] - policy_j[s,a]
    # Shape: (num_policies, 1, S, A) - (1, num_policies, S, A) -> (num_policies, num_policies, S, A)
    abs_differences: NDArray[jnp.float_] = jnp.abs(
        stacked_policies[:, jnp.newaxis, :, :] - stacked_policies[jnp.newaxis, :, :, :]
    )
    
    # Sum absolute differences along the action dimension (axis=3)
    # If actions for a state differ, sum is 2 (e.g. |1-0| + |0-1| = 2). If same, sum is 0.
    # Shape: (num_policies, num_policies, S)
    sum_abs_diff_per_state: NDArray[jnp.float_] = jnp.sum(abs_differences, axis=3)
    
    # Count number of states where policies differ.
    # A state has differing actions if sum_abs_diff_per_state for that state is > 0 (actually == 2).
    # num_differing_states[i,j] = count of states where policy i and j differ.
    # Using sum_abs_diff_per_state == 2 checks for one-hot encoded difference.
    # Total sum of differences: sum over states of (sum over actions of |pi1-pi2|)
    # This is `2 * num_states_where_actions_differ`.
    # So, if policies differ in exactly one state, this sum will be 2.
    total_sum_abs_differences: NDArray[jnp.float_] = jnp.sum(sum_abs_diff_per_state, axis=2)

    adjacency_matrix: NDArray[jnp.float_] = (total_sum_abs_differences == 2).astype(jnp.float32)
    return adjacency_matrix

def estimate_coeffs(
    basis: NDArray[jnp.float_], 
    target_vector: NDArray[jnp.float_]
) -> NDArray[jnp.float_]:
    """
    Estimates coefficients `alpha` such that `basis @ alpha ~= target_vector` using pseudo-inverse.
    This solves the least squares problem: min ||basis @ alpha - target_vector||_2^2.

    The equation is `V_basis @ alpha = V_target`.
    If basis is (S, N_basis_vectors) and target_vector is (S,),
    then alpha should be (N_basis_vectors,).
    The pseudo-inverse solution is `alpha = pinv(basis) @ target_vector`.
    The original code `np.dot(x, linalg.pinv(basis))` implies `x` is target_vector (row)
    and `pinv(basis)` is `pinv_basis` (cols, rows).
    So, `target_vector (1,S) @ pinv_basis (N_basis,S).T = target_vector (1,S) @ pinv_basis.T (S, N_basis)`
    This seems to be `target_vector @ pinv(basis).T`.
    Standard form: `coeffs = pinv(basis) @ target_vector`.
    If basis is (S, N_basis) and target_vector is (S,1) or (S,).
    pinv(basis) is (N_basis, S).
    coeffs = pinv(basis) (N_basis,S) @ target_vector (S,) -> (N_basis,). This is standard.
    The original code `np.dot(x, linalg.pinv(basis))` with x as target_vector (1,S)
    and basis (S, N_basis) means: x (1,S) @ pinv(basis) (N_basis,S) -> this is not conformable.
    If basis was (N_basis, S), then x (1,S) @ pinv(basis) (S, N_basis) -> (1, N_basis).
    Let's assume standard form: basis is (num_features, num_basis_vectors), target is (num_features,).
    Here, num_features = num_states.

    Args:
        basis: A 2D JAX array of shape (num_states, num_basis_vectors), where each
               column is a basis vector (e.g., a value function V(s)).
        target_vector: A 1D JAX array of shape (num_states,) representing the vector
                       to be approximated as a linear combination of basis vectors.

    Returns:
        A 1D JAX array of shape (num_basis_vectors,) containing the estimated coefficients `alpha`.
    """
    # Using numpy.linalg.pinv as jax.numpy.linalg.pinv might not be available in older JAX
    # or if strict numpy compatibility for pinv is desired.
    # However, for JAX arrays, jnp.linalg.pinv is preferred.
    pinv_basis: NDArray[jnp.float_] = jnp.linalg.pinv(basis)
    coefficients: NDArray[jnp.float_] = jnp.dot(pinv_basis, target_vector)
    # Original: alphas = np.dot(x, linalg.pinv(basis))
    # If x is (S,) and basis is (S, N_basis), pinv(basis) is (N_basis, S).
    # x (S,) @ pinv(basis) (N_basis,S) is not standard.
    # x (1,S) @ pinv(basis) (N_basis,S) is not standard.
    # If x (target_vector) is (S,) and basis is (S, N_basis_vectors),
    # then coefficients = jnp.linalg.lstsq(basis, target_vector)[0] is another way.
    # The most common formulation is `pinv(A) b`.
    return coefficients

def mse(vector_x: NDArray[jnp.float_], vector_y: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
    """
    Calculates the Mean Squared Error (MSE) between two vectors.
    Note: This implementation returns the Sum of Squared Errors (SSE), not mean.
    For true MSE, divide by the number of elements.

    Args:
        vector_x: The first vector.
        vector_y: The second vector, must have the same shape as vector_x.

    Returns:
        The sum of squared differences between vector_x and vector_y (a scalar JAX array).
    """
    return jnp.sum(jnp.square(vector_x - vector_y))

def sparse_coeffs(
    basis: NDArray[jnp.float_], 
    target_vector: NDArray[jnp.float_], 
    regularization_strength: float = 1e-6, 
    learning_rate: float = 1e-1, 
    initial_coefficients_params: Optional[NDArray[jnp.float_]] = None,
    momentum_coeff: float = 0.9,
    max_iterations: int = 1000, # Added for utils.solve
    tolerance: float = 1e-6     # Added for utils.solve
) -> NDArray[jnp.float_]:
    """
    Finds sparse coefficients `alpha` that approximate `target_vector` as a linear combination
    of `basis` vectors, with an L1-like penalty (entropy regularization on softmax output).

    Solves the optimization problem:
    min_x || basis @ softmax(x) - target_vector ||_2^2 + regularization_strength * entropy(softmax(x))
    where `alpha = softmax(x)`. The softmax ensures `alpha` are positive and sum to 1 (convex combination),
    and entropy term promotes sparsity in `alpha`.

    Args:
        basis: 2D JAX array of shape (num_features, num_basis_vectors), e.g. (S, N_det_policies).
        target_vector: 1D JAX array of shape (num_features,), e.g. (S,).
        regularization_strength: Strength of the entropy regularization term (gamma).
        learning_rate: Learning rate for the gradient descent optimizer.
        initial_coefficients_params: Optional initial parameters `x` for `softmax(x)`.
                                     If None, initialized to small random values. Shape (num_basis_vectors,).
        momentum_coeff: Momentum coefficient for the optimizer.
        max_iterations: Maximum iterations for the solver.
        tolerance: Tolerance for convergence for the solver.


    Returns:
        A 1D JAX array of shape (num_basis_vectors,) representing the optimized softmax parameters `x`
        (not the coefficients `alpha = softmax(x)` directly, but the parameters that produce them).
    """
    assert basis.shape[0] == target_vector.shape[0], "Basis and target vector feature dimensions must match."
    num_basis_vectors: int = basis.shape[1]

    def sparse_loss(params_x: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        # Coefficients alpha are derived from params_x via softmax
        alpha_coeffs: NDArray[jnp.float_] = utils.softmax(params_x)  # Convex combination
        reconstruction_error: NDArray[jnp.float_] = mse(jnp.dot(basis, alpha_coeffs), target_vector)
        entropy_term: NDArray[jnp.float_] = utils.entropy(alpha_coeffs) # utils.entropy might need positive inputs
        return reconstruction_error + regularization_strength * entropy_term

    # Gradient of the loss function with respect to params_x
    dL_dparams_x = grad(sparse_loss)
    
    # Jit the update function for performance
    @jit
    def update_fn_momentum(params_x_and_momentum: Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]) -> Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]:
        params_x, momentum_vector = params_x_and_momentum
        gradient: NDArray[jnp.float_] = dL_dparams_x(params_x)
        
        # Standard momentum update
        new_momentum: NDArray[jnp.float_] = momentum_coeff * momentum_vector - learning_rate * gradient
        new_params_x: NDArray[jnp.float_] = params_x + new_momentum
        return new_params_x, new_momentum

    if initial_coefficients_params is None:
        # Small random initialization for x (parameters for softmax)
        # Using numpy.random for consistency with original, JAX random is different.
        params_x_init: NDArray[jnp.float_] = 1e-3 * rnd.standard_normal((num_basis_vectors,))
    else:
        params_x_init = initial_coefficients_params

    # Initial state for the solver (parameters_x, momentum_vector)
    initial_solver_state: Tuple[NDArray[jnp.float_], NDArray[jnp.float_]] = (
        params_x_init, 
        jnp.zeros_like(params_x_init)
    )
    
    # Assuming search_spaces.momentum_bundler is compatible or this is how it's structured
    # The original utils.solve might need max_iter and tol.
    # The `momentum_bundler` seems to be a simple way to pack the update logic for `utils.solve`.
    # Let's assume `utils.solve` takes an update function and an initial state.
    # Bundler might simplify this to just `update_fn(params_x)` if momentum is handled inside.
    # Based on original: `search_spaces.momentum_bundler(update_fn, 0.9)` where update_fn takes only x.
    # This implies the bundler wraps `update_fn` to manage momentum internally.
    # Let's adjust `update_fn` to match that expectation if `momentum_bundler` handles state.
    
    # Re-defining update_fn if momentum_bundler expects a simpler signature (manages momentum itself)
    # This depends on the implementation of search_spaces.momentum_bundler
    # For now, assume `utils.solve` can handle the tuple state with `update_fn_momentum`.
    
    # The original `utils.solve` might be a simple loop. Adding max_iter and tol.
    # If `utils.solve` is a generic solver, it might take `(value, iteration_count)`
    # and stop based on `iteration_count` or change in `value`.
    # For now, using the structure as provided.
    
    # output_trajectory is a list of (params_x_t, momentum_t) tuples over iterations
    output_trajectory: List[Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]] = utils.solve(
        update_fn_momentum, 
        initial_solver_state,
        max_iter=max_iterations, # Added
        tol=tolerance          # Added
    )
    
    # Unzip the trajectory of parameters and momentums
    final_params_x_trajectory, _ = zip(*output_trajectory)
    
    # Return the last set of parameters x
    return final_params_x_trajectory[-1]
