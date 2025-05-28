from typing import Tuple, List, Any, Callable
import numpy # Used for linalg.pinv if jax.numpy.linalg.pinv is not directly used.
from numpy import linalg # Specifically for pinv if needed.
import numpy.random as rnd # For random number generation. Consider jax.random for pure JAX.

import jax.numpy as jnp
from jax import grad, jit
from numpy.typing import NDArray # For type hinting.

import mdp.utils as utils

# Note on JAX vs NumPy:
# This file mixes `numpy` (imported as numpy, and its linalg) and `jax.numpy` (imported as np, then changed to jnp).
# It's generally better to stick to one for core computations if possible, ideally JAX if JAX features like grad/jit are used.
# `rnd` is numpy.random. For JAX, `jax.random` with explicit PRNGKeys is the standard.
# I will use `jnp` for JAX arrays and operations, and `numpy` for things like `linalg.pinv` if it's specifically chosen over `jnp.linalg.pinv`.

def rnd_lmdp(num_states: int, num_actions: int) -> Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]:
    """
    Generates a random "Linear MDP" (LMDP) structure.
    The term LMDP here refers to a specific parameterization (p, q) rather than a standard MDP.
    'p' can be thought of as an uncontrolled transition matrix (state-to-state).
    'q' can be thought of as a pseudo-reward or state potential.

    Args:
        num_states: The number of states.
        num_actions: The number of actions (Note: num_actions is not directly used in this function
                     to define the shape of p or q, which might be an oversight if 'p' or 'q'
                     are expected to relate to actions in some contexts where this function is used).

    Returns:
        A tuple (p, q):
            - p: A JAX array of shape (num_states, num_states), representing column-stochastic
                 transitions (p[:, i] sums to 1). This is an "uncontrolled" transition matrix.
            - q: A JAX array of shape (num_states, 1), representing pseudo-rewards or potentials.
    """
    # Using numpy.random for generation, then converting to JAX array.
    p_np: NDArray[numpy.float_] = rnd.random((num_states, num_states))
    p_np = p_np / p_np.sum(axis=0) # Normalize columns to sum to 1
    
    q_np: NDArray[numpy.float_] = rnd.random((num_states, 1))
    
    return jnp.array(p_np), jnp.array(q_np)

def mdp_encoder(
    P_mdp: NDArray[jnp.float_], 
    r_mdp: NDArray[jnp.float_]
) -> Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]:
    """
    Encodes a standard MDP (P, r) into an LMDP representation (p, q) for each state.
    This encoding is based on principles from Todorov's work on Linearly Solvable MDPs,
    where `p` represents uncontrolled dynamics and `q` represents a pseudo-reward or potential.
    The process involves solving a system of equations for each state to find its
    corresponding `p(s'|s)` and `q(s)`.

    Args:
        P_mdp: The transition probability tensor of the original MDP, shape (S, A, S').
               P_mdp[s, a, s_prime] is P(s_prime | s, a).
               Note: The original code comment `P[:, idx_x, :]` implies P is (S', S, A).
               Assuming standard (S, A, S') and adjusting indexing. If original was (S',S,A) use P_mdp.transpose(2,1,0)
        r_mdp: The reward matrix of the original MDP, shape (S, A). r_mdp[s, a] is R(s, a).

    Returns:
        A tuple (p_encoded, q_encoded):
            - p_encoded: The uncontrolled transition matrix, shape (S, S).
                         p_encoded[:, s] is the distribution p(s'|s).
            - q_encoded: The pseudo-reward vector, shape (S,).
    """
    num_states: int = P_mdp.shape[0]

    def embed_state(state_index: int) -> Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]:
        """
        Inner function to compute p(.|s) and q(s) for a single state `s` (state_index).
        It solves a system derived from the LMDP conditions:
        - P(s'|s,a) = u(s'|a) * p(s'|s)  (where u is control, p is uncontrolled dynamics)
        - R(s,a) = q(s) - KL( u(.|a)p(.|s) || p(.|s) )  (simplified, involves control cost)
        The actual derivation from Todorov (2009) is more nuanced, often involving:
        R(s,a) + log p(s'|s) = log P(s'|s,a) - log u(s'|a) + q(s)
        This function implements a specific formulation from the paper's supplement.

        Args:
            state_index: The index of the current state `s` for which to compute p and q.

        Returns:
            A tuple (p_s, q_s_scalar):
                - p_s: Uncontrolled transition distribution from state `s`, p(s'|s). Shape (S,).
                - q_s_scalar: Pseudo-reward for state `s`. Shape (1,) then squeezed.
        """
        # Rewards for current state, all actions: r(s,a) for a in A. Shape (A,)
        # Original P indexing P[:, idx_x, :] means (S_prime, S, A)
        # If P_mdp is (S, A, S'), then P_mdp[state_index, :, :] is (A, S')
        # And P_mdp[state_index, a, :] is P(s'|s,a)
        # D should be P(s'|s,a) for fixed s. (A, S') if rows are actions, (S', A) if cols are actions.
        # Original D = P[:, idx_x, :] implies P is (S', S, A), so D is (S', A).
        # Let's assume P_mdp(S, A, S'), so for fixed s=state_index: P_s_asprime = P_mdp[state_index, :, :] (A, S')
        # To match D shape (S', A) from original, this would be P_mdp[state_index, :, :].T
        
        # b_vector's role: related to rewards r(s,a) for fixed s. Shape (A,)
        b_vector: NDArray[jnp.float_] = r_mdp[state_index, :]
        
        # D_matrix's role: P(s'|s,a) for fixed s.
        # Original D = P[:, idx_x, :] implies shape (S', A).
        # If P_mdp is (S,A,S'), then for s=state_index, P(s'|s,a) is P_mdp[state_index, a, s'].
        # To get (S', A) from (S,A,S'), it's P_mdp[state_index, :, :].T
        D_matrix: NDArray[jnp.float_] = P_mdp[state_index, :, :].T # Shape (S', A)

        # System to solve (from paper supplement): c_vector related to log u - q
        # D_matrix.T @ (q_s * 1_vector - m_vector) = b_vector (this is not quite it)
        # The paper uses: b_a = r(s,a) and D_as' = P(s'|s,a).
        # It solves for m_s'(vector) and q_s (scalar) from:
        # r(s,a) = sum_{s'} P(s'|s,a) * (q_s - m_{s'})
        # r(s,a) = q_s * sum_{s'} P(s'|s,a) - sum_{s'} P(s'|s,a) * m_{s'}
        # r(s,a) = q_s - sum_{s'} P(s'|s,a) * m_{s'}  (since sum_{s'} P(s'|s,a) = 1)
        # This is a linear system for q_s and m_{s'}.
        # The code `c = np.dot(b, linalg.pinv(D))` implies `c D = b` if c is row vec, or `D.T c.T = b.T`
        # If D is (S',A), pinv(D) is (A,S'). b is (A,). So c is (S',).  c = b @ pinv(D)
        # This means c is like `q_s * 1 - m` from paper.
        c_vector: NDArray[jnp.float_] = jnp.dot(b_vector, numpy.linalg.pinv(D_matrix)) # Use numpy.linalg.pinv for now. (S',)

        # The derivation for q_s (scalar potential/pseudo-reward for state s) and
        # m_s_prime (related to uncontrolled dynamics p(s'|s)) from c_vector:
        # q_s = log sum_{s'} exp(c_{s'})  (This is a LogSumExp, often used for normalization)
        # m_{s'} = c_{s'} - q_s
        # p(s'|s) = exp(m_{s'}) = exp(c_{s'} - q_s) = exp(c_{s'}) / exp(q_s)
        # This ensures p(s'|s) sums to 1 because exp(q_s) is the sum of exp(c_{s'}).
        q_s_scalar: NDArray[jnp.float_] = jnp.log(jnp.sum(jnp.exp(c_vector))) 
        
        m_s_prime_vector: NDArray[jnp.float_] = c_vector - q_s_scalar 
        p_s_prime_given_s: NDArray[jnp.float_] = jnp.exp(m_s_prime_vector) # This is p(s'|s). Shape (S',)

        # Normalization check for p(s'|s)
        # This check should ideally pass due to the LSE formulation.
        # The original code had a check that was specific to its p.sum(0) if p was S'xS
        # Here p_s_prime_given_s is a vector for a single s, so sum over S'
        if not jnp.isclose(jnp.sum(p_s_prime_given_s), 1.0, atol=1e-3):
            # print(f"Warning: p(s'|s={state_index}) does not sum to 1: {jnp.sum(p_s_prime_given_s)}")
            # raise ValueError(f'p(s\'|s={state_index}) is not normalised')
            # It might be better to re-normalize explicitly if small errors occur, or check tolerance.
             p_s_prime_given_s = p_s_prime_given_s / jnp.sum(p_s_prime_given_s)


        return p_s_prime_given_s, jnp.array([q_s_scalar]) # q_s_scalar is already scalar, wrap for consistency if needed

    # Apply embed_state for each state
    # `embedded_p_and_q_list` will be a list of tuples (p_s_vector, q_s_scalar_array)
    embedded_p_and_q_list: List[Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]] = [
        embed_state(i) for i in range(num_states)
    ]
    
    # Unzip the list of tuples
    # p_vectors will be a tuple of S, arrays, each of shape (S,)
    # q_scalars will be a tuple of S, arrays, each of shape (1,)
    p_vectors_tuple, q_scalars_tuple = zip(*embedded_p_and_q_list)
    
    # Stack p_vectors: each p_s_vector becomes a column in the final p_encoded matrix.
    # So, p_encoded[s', s] = p(s'|s). Shape (S, S).
    p_encoded: NDArray[jnp.float_] = jnp.stack(p_vectors_tuple, axis=1)
    
    # Stack q_scalars and squeeze to get a vector of shape (S,).
    q_encoded: NDArray[jnp.float_] = jnp.squeeze(jnp.stack(q_scalars_tuple, axis=0)) # Stack along new axis, then squeeze.
    
    return p_encoded, q_encoded


def KL(P_dist: NDArray[jnp.float_], Q_dist: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
    """
    Computes the Kullback-Leibler (KL) divergence D_KL(P || Q).
    KL(P || Q) = sum_i P[i] * log(P[i] / Q[i])
               = sum_i P[i] * (log P[i] - log Q[i])
               = - sum_i P[i] * (log Q[i] - log P[i])
               = - sum_i P[i] * log(Q[i] / P[i])
    The original formula: -np.sum(P*np.log((Q+1e-8)/(P+1e-8))) matches this.
    Small epsilon (1e-8) is added for numerical stability to avoid log(0) or division by zero.

    Args:
        P_dist: The "true" probability distribution, a 1D JAX array. Sums to 1.
        Q_dist: The "approximating" probability distribution, a 1D JAX array. Sums to 1.
                Must have the same shape as P_dist.

    Returns:
        A scalar JAX array representing the KL divergence.
    """
    epsilon: float = 1e-8
    # Ensure inputs are JAX arrays for JAX operations
    P_dist_jnp = jnp.asarray(P_dist)
    Q_dist_jnp = jnp.asarray(Q_dist)
    
    return -jnp.sum(P_dist_jnp * jnp.log((Q_dist_jnp + epsilon) / (P_dist_jnp + epsilon)))

def CE(P_dist: NDArray[jnp.float_], Q_dist: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
    """
    Computes the Cross-Entropy H(P, Q) = - sum_i P[i] * log(Q[i]).
    The original formula `np.sum(P*np.log(Q+1e-8))` is -H(P,Q) if P is probability.
    If it's just sum P_i log Q_i, it's related to cross-entropy.
    Standard cross-entropy is H(P,Q) = - sum P(x) log Q(x).
    The provided code `np.sum(P*np.log(Q+1e-8))` is sum_i P[i] log Q[i].
    This is not exactly cross-entropy but a component of it or KL divergence.
    Let's assume it's calculating sum_i P[i] * log Q[i].

    Args:
        P_dist: A distribution or weights, a 1D JAX array.
        Q_dist: A distribution (values typically > 0 for log), a 1D JAX array.
                Must have the same shape as P_dist.

    Returns:
        A scalar JAX array representing sum_i P_dist[i] * log(Q_dist[i] + epsilon).
    """
    epsilon: float = 1e-8
    P_dist_jnp = jnp.asarray(P_dist)
    Q_dist_jnp = jnp.asarray(Q_dist)
    return jnp.sum(P_dist_jnp * jnp.log(Q_dist_jnp + epsilon))

@jit
def linear_bellman_operator(
    p_uncontrolled: NDArray[jnp.float_], 
    q_pseudo_reward: NDArray[jnp.float_], 
    z_exp_value: NDArray[jnp.float_], 
    discount: float
) -> NDArray[jnp.float_]:
    """
    Applies the linear Bellman operator found in Linearly Solvable MDPs.
    The operator is defined as: z_next(s) = exp(q(s)) * sum_{s'} p(s'|s) * z_exp_value(s')^discount
    where `z_exp_value(s)` is often e^{value_function(s)}.

    Args:
        p_uncontrolled: The uncontrolled transition matrix, shape (S, S).
                        p_uncontrolled[s', s] is p(s'|s). (Note: original was p[s,s'])
                        If p_uncontrolled is (S_rows=dest_state, S_cols=source_state)
        q_pseudo_reward: The pseudo-reward vector, shape (S,) or (S,1).
        z_exp_value: The current exponentiated value function, z(s) = e^{V(s)}, shape (S,) or (S,1).
        discount: The discount factor (gamma).

    Returns:
        The next exponentiated value function, z_next(s), shape (S,).
    """
    # Ensure q is 1D (S,) for diagonal matrix, and z is 1D (S,) for power.
    q_squeezed: NDArray[jnp.float_] = jnp.squeeze(q_pseudo_reward)
    z_squeezed: NDArray[jnp.float_] = jnp.squeeze(z_exp_value)

    # Q_diag_exp_q = diag(e^q(s))
    Q_diag_exp_q: NDArray[jnp.float_] = jnp.diag(jnp.exp(q_squeezed))
    
    # p_uncontrolled is likely p(s_prime | s_source), so columns sum to 1. (S_prime_rows, S_source_cols)
    # (Q @ p) part: (S,S) @ (S,S) -> (S,S). Let this be M.
    # M[s, s_source] = exp(q[s]) * p[s, s_source] (if p is dest_state_rows, source_state_cols and q related to dest_state)
    # Or, M[s_dest, s_source] = exp(q[s_source]) * p[s_dest | s_source] if Q_diag is based on source state q.
    # Given `np.dot(Q, p)`: if Q is diag(exp(q[s_row])), and p is p[s_row, s_col],
    # then (Q @ p)[i,j] = Q[i,i] * p[i,j] = exp(q[i]) * p[i,j]. This means p is p(s_col | s_row). (Rows sum to 1).
    # If p is p(s'|s) where columns sum to 1 (p[s',s]), then (Q @ p)[s',s] = exp(q[s']) * p[s',s].
    # The equation z(s) = exp(q(s)) * E_{s'~p(s'|s)} [z(s')^gamma] means:
    # z_next_s = exp(q_s) * sum_{s'} p(s'|s) * (z_s')^gamma
    # This implies: Q_diag should be exp(q_s) for the specific s.
    # And sum over s' is a dot product with a column of p: p[:, s].
    # So, ( (z_s')^gamma ) @ p[:,s] gives the sum. Then multiply by exp(q_s).
    # (z_s')^gamma is a vector. Let it be z_pow.
    # (Q @ p) @ z_pow: (S,S) @ (S,S) @ (S,) -> (S,)
    # (Q @ p)[i,j] = exp(q[i]) p[i,j] (assuming p is row-stochastic: p[i,j] = p(j|i))
    # Then ((Q @ p) @ z_pow)[i] = sum_j (exp(q[i]) p[i,j]) * z_pow[j]
    # This is exp(q[i]) * sum_j p(j|i) * z_pow[j]. This matches the formula.
    # So, p must be row-stochastic: p[i,j] means p(j|i).
    
    z_pow_gamma: NDArray[jnp.float_] = z_squeezed ** discount
    # Q_diag_exp_q is (S,S), p_uncontrolled is (S,S), z_pow_gamma is (S,)
    # result = Q_diag_exp_q @ p_uncontrolled @ z_pow_gamma
    # This is equivalent to:
    # term1 = Q_diag_exp_q @ p_uncontrolled # (S,S)
    # result = term1 @ z_pow_gamma # (S,)
    # Or more directly:
    expected_sum_term = jnp.dot(p_uncontrolled, z_pow_gamma) # sum_{s'} p(s'|s) * z(s')^gamma, if p is p(s'|s_row)
                                                           # If p is p(s'|s_col) (col stochastic), then p.T @ z_pow_gamma
    # Assuming p is row-stochastic (p[s, s_prime] = p(s_prime|s))
    return jnp.exp(q_squeezed) * expected_sum_term


@jit
def linear_value_functional(
    p_uncontrolled: NDArray[jnp.float_], 
    q_pseudo_reward: NDArray[jnp.float_], 
    u_optimal_control: NDArray[jnp.float_], 
    discount: float
) -> NDArray[jnp.float_]:
    """
    Computes the value function V(s) for a Linearly Solvable MDP (LMDP) given optimal control `u`.
    The value function is defined as:
    V(s) = q(s) + sum_{s'} u(s'|s) * (log p(s'|s) - log u(s'|s)) + discount * sum_{s'} u(s'|s) * V(s')
         = q(s) - KL(u(.|s) || p(.|s)) + discount * E_{s'~u(.|s)} [V(s')]
    This is a system of linear equations for V(s), which can be solved.

    Args:
        p_uncontrolled: Uncontrolled transition matrix, shape (S, S). p_uncontrolled[s', s] is p(s'|s).
                        Assumed column-stochastic (columns sum to 1).
        q_pseudo_reward: Pseudo-reward vector, shape (S,) or (S,1).
        u_optimal_control: Optimal control dynamics (policy-conditioned transitions), shape (S, S).
                           u_optimal_control[s', s] is u*(s'|s). Assumed column-stochastic.
        discount: Discount factor (gamma).

    Returns:
        The value function V(s) as a 1D JAX array of shape (S,).
    """
    num_states: int = p_uncontrolled.shape[0]
    q_squeezed: NDArray[jnp.float_] = jnp.squeeze(q_pseudo_reward)

    # r_lmdp[s] = q(s) + sum_{s'} u(s'|s) * log(p(s'|s) / u(s'|s))
    #           = q(s) - KL(u(.|s) || p(.|s)) for each state s.
    # This requires u and p to be column-stochastic for KL interpretation.
    # u[s',s] = u(s'|s), p[s',s] = p(s'|s)
    # log_term = log (p/u) needs element-wise operations.
    epsilon: float = 1e-8 # To prevent log(0) or division by zero.
    
    # KL_divergence_per_state = sum_{s'} u(s'|s) * log(u(s'|s) / p(s'|s))
    # The term in the paper is sum u log(p/u) = -KL(u||p)
    # So, r_lmdp_s = q_s - KL(u(.|s) || p(.|s))
    # Original code: r_lmdp = q + np.sum(u*np.log(p/u), axis=0)
    # If u and p are (S_dest_rows, S_source_cols), then sum(axis=0) means sum over s_dest.
    # This gives one r_lmdp value per source state s.
    # r_lmdp[s] = q[s] (assuming q relates to source state s) + sum_{s'} u[s',s] * log(p[s',s]/u[s',s])
    
    # Ensure u and p are column-stochastic for correct KL interpretation.
    # (This should be guaranteed by lmdp_solver for u, and rnd_lmdp for p).
    log_ratio: NDArray[jnp.float_] = jnp.log((p_uncontrolled + epsilon) / (u_optimal_control + epsilon))
    effective_reward_lmdp: NDArray[jnp.float_] = q_squeezed + jnp.sum(u_optimal_control * log_ratio, axis=0) # Sum over s' (rows)

    # System: V = r_lmdp + discount * u^T @ V  (if u is u(s'|s_source_col))
    # (I - discount * u^T) @ V = r_lmdp
    # V = inv(I - discount * u^T) @ r_lmdp
    # The original code uses `discount*u` which means u must be `u(s_next_row | s_current_col)`.
    # If u is u(s'|s) where columns are s (source), then u.T is needed for standard Bellman form.
    # u is (S_dest_rows, S_source_cols). `u.T` would be (S_source_rows, S_dest_cols).
    # V_s = r_s + gamma * sum_s' u(s'|s) V_s' -> V = r + gamma * u.T @ V
    # (I - gamma * u.T) V = r
    identity_matrix: NDArray[jnp.float_] = jnp.eye(num_states)
    matrix_to_invert: NDArray[jnp.float_] = identity_matrix - discount * u_optimal_control.T
    
    state_values: NDArray[jnp.float_] = jnp.linalg.solve(matrix_to_invert, effective_reward_lmdp)
    return state_values

def lmdp_solver(
    p_uncontrolled: NDArray[jnp.float_], 
    q_pseudo_reward: NDArray[jnp.float_], 
    discount: float,
    max_iterations: int = 1000, # Added for utils.solve
    tolerance: float = 1e-7     # Added for utils.solve
) -> Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]:
    """
    Solves the Linearly Solvable MDP (LMDP) problem.
    This involves finding the exponentiated value function `z(s) = exp(V(s))` by iterating:
    z_next(s) = exp(q(s)) * sum_{s'} p(s'|s) * z(s')^discount
    until convergence. Then, the optimal control `u*(s'|s)` and standard value `V(s)` are derived.

    Args:
        p_uncontrolled: Uncontrolled transition matrix, shape (S, S).
                        p_uncontrolled[s_row, s_col] is p(s_col | s_row) (row-stochastic).
                        Or, p_uncontrolled[s_col, s_row] is p(s_col | s_row) (col-stochastic for Bellman op).
                        The `linear_bellman_operator` assumes p is row-stochastic p(next_state | current_state).
        q_pseudo_reward: Pseudo-reward vector, shape (S,) or (S,1).
        discount: Discount factor (gamma).
        max_iterations: Max iterations for the solver.
        tolerance: Convergence tolerance for z.


    Returns:
        A tuple (u_optimal, V_optimal):
            - u_optimal: Optimal control dynamics (policy-conditioned transitions),
                         shape (S, S). u_optimal[s', s] = u*(s'|s) (column-stochastic).
            - V_optimal: Optimal value function V*(s), shape (S,).
    """
    # Note: The comment "BUG doesnt work for large discounts: 0.999" from original code.
    num_states: int = p_uncontrolled.shape[0]

    # Initial guess for z (e.g., ones, corresponding to V=0)
    # Original init: np.ones((p.shape[-1], 1)) - assumes p is (S, S_something), takes last dim.
    # If p is (S,S), then p.shape[-1] is S.
    z_initial_guess: NDArray[jnp.float_] = jnp.ones((num_states, 1))

    # Define the update function for z using the linear Bellman operator
    update_fn_z: Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]] = \
        lambda z_current: linear_bellman_operator(p_uncontrolled, q_pseudo_reward, z_current, discount)
    
    # Solve for z using fixed-point iteration
    # utils.solve returns a list of iterates. The last one is the converged solution.
    z_converged_trajectory: List[NDArray[jnp.float_]] = utils.solve(update_fn_z, z_initial_guess, max_iter=max_iterations, tol=tolerance)
    z_converged: NDArray[jnp.float_] = jnp.squeeze(z_converged_trajectory[-1]) # Shape (S,)
    
    # Optimal value function V*(s) = log z*(s)
    V_optimal: NDArray[jnp.float_] = jnp.log(z_converged) # Shape (S,)

    # Calculate the optimal control u*(s'|s)
    # G(s) = sum_{s'} p(s'|s) * z_converged(s')  (G is a vector of expected future z values from each state s)
    # If p_uncontrolled[row, col] = p(col | row): (row-stochastic)
    # G_s = sum_{s_prime} p_uncontrolled[s, s_prime] * z_converged[s_prime] = p_uncontrolled @ z_converged
    G_vector: NDArray[jnp.float_] = jnp.dot(p_uncontrolled, z_converged) # Shape (S,)
    
    # u*(s'|s) = p(s'|s) * z_converged(s') / G(s)
    # This requires element-wise multiplication of p(s'|s) with z_converged(s'), then normalization by G(s).
    # If p_uncontrolled[s', s_row] = p(s' | s_row) (row-stochastic), then:
    # u_optimal[s', s_row] = (p_uncontrolled[s', s_row] * z_converged[s']) / G_vector[s_row]
    # This means ( p_uncontrolled * z_converged_broadcasted_column_wise ) / G_vector_broadcasted_row_wise
    # If p_uncontrolled is (S_dest_rows, S_source_cols) (col-stochastic, p(s'|s)):
    # G_s = sum_{s'} p[s',s] z[s'] = p.T @ z (if z is col vec)
    # Original G = np.einsum('ij,i->j', p, z) -> if p is (S',S) and z is (S'), G is (S,)
    # This means p[s',s] = p(s' from s), and z is z(s'). Sum over s'. G[s] = sum_s' p[s',s]*z[s'].
    # This is p.T @ z if z is a column vector. If z is (S,), this is correct for p(s_row=s', s_col=s).
    # Let's assume p is (S_rows=destination, S_cols=source), so p[:,s] is a distribution over next states from s.
    # G_s = sum_{s'} p[s', s] * z_converged[s']
    G_vector_col_stochastic_p: NDArray[jnp.float_] = jnp.dot(z_converged, p_uncontrolled) # z.T @ p -> (S,)
    
    # u[s', s] = p[s', s] * z_converged[s'] / G_vector_col_stochastic_p[s]
    # u_optimal = (p_uncontrolled * z_converged[:, jnp.newaxis]) / G_vector_col_stochastic_p[jnp.newaxis, :]
    # This assumes z_converged is (S_dest,), G is (S_source,).
    # Corrected based on original: u = p * z[:, np.newaxis] / G[np.newaxis, :]
    # If p is (S_dest_rows, S_source_cols), z is (S_dest_rows), G is (S_source_cols)
    # z[:, np.newaxis] makes z (S_dest_rows, 1). p * z broadcast -> (S_dest, S_source)
    # G[np.newaxis, :] makes G (1, S_source). Result is (S_dest, S_source)
    # This matches u[s', s] = u*(s'|s)
    u_optimal: NDArray[jnp.float_] = (p_uncontrolled * z_converged[:, jnp.newaxis]) / G_vector_col_stochastic_p[jnp.newaxis, :]
    
    return u_optimal, V_optimal

def lmdp_decoder(
    u_target_dynamics: NDArray[jnp.float_], 
    P_original_mdp: NDArray[jnp.float_], 
    learning_rate: float = 10.0,
    max_iterations: int = 1000, # Added for utils.solve
    tolerance: float = 1e-6     # Added for utils.solve
) -> NDArray[jnp.float_]:
    """
    Decodes a target control dynamic `u_target_dynamics` into a policy `pi` for the original MDP.
    It optimizes the parameters of a softmax policy `pi(a|s) = softmax(logits(s,a))`
    such that the policy-conditioned transition matrix `P_pi(s'|s) = sum_a pi(a|s)P(s'|s,a)`
    is close to `u_target_dynamics(s'|s)`. Closeness is measured by KL divergence.

    Args:
        u_target_dynamics: Target optimal control dynamics, shape (S, S).
                           u_target_dynamics[s', s] = u*(s'|s) (column-stochastic).
        P_original_mdp: Transition probability tensor of the original MDP, shape (S, A, S').
                        P_original_mdp[s, a, s_prime] = P(s_prime | s, a).
                        (Note: original code uses P.shape[0] for S, P.shape[-1] for A in init,
                         and einsum 'ijk,jk->ij' for P, pi.
                         If P is (S_orig, A_orig, S_prime_orig) and pi is (S_orig, A_orig),
                         then P_pi is (S_orig, S_prime_orig). This means P_pi[s,s'] = sum_a P[s,a,s']pi[s,a].
                         This matches typical P_pi(s'|s).
                         u_target_dynamics should be (S,S) where u[s',s] = u(s'|s).
                         The KL divergence sum(u * log(u/P_pi)) implies element-wise comparison.
                         If u is (S_dest, S_source) and P_pi is (S_source, S_dest), need transpose.
                         Assuming u and P_pi are (S_dest_rows, S_source_cols) for KL.
                         The einsum 'ijk,jk->ij' for P(S,A,S'), pi(S,A) gives P_pi(S,S') (row s, col s').
                         So P_pi[s,s'] = P_pi(s'|s).
                         Then u_target_dynamics should also be u(s'|s_row).
                         This means u_target_dynamics.T if it was (S_dest, S_source_col).
                         Let's assume inputs are consistently u(s'|s_row_idx) and P_pi(s'|s_row_idx).
        learning_rate: Learning rate for gradient descent.
        max_iterations: Max iterations for the solver.
        tolerance: Convergence tolerance for policy logits.

    Returns:
        The learned policy `pi` as a JAX array of shape (S, A), where `pi[s,a]` is
        the probability of taking action `a` in state `s`.
    """
    num_states: int = P_original_mdp.shape[0]
    num_actions: int = P_original_mdp.shape[1] # Assuming P is (S,A,S')

    def loss_fn(policy_logits: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        policy: NDArray[jnp.float_] = utils.softmax(policy_logits, axis=1) # Softmax over actions for each state
        # P_policy[s, s'] = sum_a P_original_mdp[s, a, s'] * policy[s, a]
        P_policy: NDArray[jnp.float_] = jnp.einsum('sak,sa->sk', P_original_mdp, policy) # (S,S')
        
        # KL(u_target || P_policy). u_target needs to be u(s'|s_row)
        # If u_target_dynamics is u(s'_col | s_row), then it's (S_rows=s', S_cols=s_row).
        # And P_policy is (S_rows=s, S_cols=s').
        # Need them to align. If u is (S_dest, S_source_col) and P_pi is (S_source_row, S_dest_col)
        # then KL(u.T || P_pi) or KL(u || P_pi.T).
        # Original: sum(u * log(u/P_pi)). This implies u and P_pi are element-wise comparable.
        # Let's assume u_target_dynamics[s,s'] = u(s'|s) and P_policy[s,s'] = P_pi(s'|s)
        # This means rows are current states, columns are next states.
        kl_divergence: NDArray[jnp.float_] = jnp.sum(
            u_target_dynamics * jnp.log(u_target_dynamics / (P_policy + 1e-8) + 1e-8)
        ) # Sum over all elements
        return kl_divergence

    dL_dlogits: Callable = jit(grad(loss_fn))
    
    def update_fn_logits(current_logits: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        gradient: NDArray[jnp.float_] = dL_dlogits(current_logits)
        return current_logits - learning_rate * gradient

    # Initialize policy logits
    # Original P.shape[0] (num_states), P.shape[-1] (num_actions, if P was S,S,A)
    # If P is (S,A,S'), then P.shape[1] is num_actions.
    initial_logits: NDArray[jnp.float_] = rnd.standard_normal((num_states, num_actions))
    
    # Solve for optimal logits
    final_logits_trajectory: List[NDArray[jnp.float_]] = utils.solve(
        update_fn_logits, 
        initial_logits,
        max_iter=max_iterations,
        tol=tolerance
    )
    optimal_logits: NDArray[jnp.float_] = final_logits_trajectory[-1]
    
    return utils.softmax(optimal_logits, axis=1)


def option_transition_fn(
    P_base_actions: NDArray[jnp.float_], 
    num_steps_k: int
) -> NDArray[jnp.float_]:
    """
    Computes transition probabilities for k-step options.
    An option here is a sequence of k base actions.
    The resulting tensor P_options[s, s', option_idx] gives the probability of
    transitioning from state s to state s' by executing the sequence of actions
    corresponding to option_idx.

    Args:
        P_base_actions: Transition probability tensor for base actions, shape (S, A, S').
                        P_base_actions[s, a, s'] = P(s'|s,a).
        num_steps_k: The length of action sequences (options).

    Returns:
        A 3D JAX array P_all_options[s, s', option_index] where option_index enumerates
        all possible sequences of k actions. Shape (S, S, A^k) if structured simply,
        or (S, S, A + A^2 + ... + A^k) if concatenating options of different lengths up to k.
        The original code structure `np.concatenate(Ps, axis=-1)` and
        `n_options = sum([n_actions**(i+1) for i in range(k)])` implies the latter:
        options are sequences of length 1, OR length 2, ..., OR length k.
    """
    num_states: int = P_base_actions.shape[0]
    num_base_actions: int = P_base_actions.shape[1]

    # List to store P_len_i[s, s', specific_sequence_of_length_i]
    # P_len_1 is P_base_actions, reshaped to (S, S, A) by choosing one s' for each P[s,a,s']
    # This seems problematic. P(s'|s,a_1,...,a_i) is needed.
    # P_base_actions is (S,A,S'). Transpose to (S,S',A) for easier einsum? (s,s',a)
    # P_base_actions_T = P_base_actions.transpose((0,2,1)) # (S, S', A)
    
    # Ps_list will store transition matrices for options of length 1, 2, ..., k.
    # P_len_1_options should be P_base_actions itself, mapping (s, option_len_1) -> s'
    # Shape (S, S', A) - P_options_len1[s, s', a1_idx] = P(s' | s, a1)
    Ps_list: List[NDArray[jnp.float_]] = [P_base_actions.transpose(0,2,1)] # P(s,s',a)

    # For options of length i > 1
    # P_len_i(s, s', (a1,...,ai)) = sum_{s_intermediate} P_len_{i-1}(s, s_intermediate, (a1,...,a_{i-1})) * P(s' | s_intermediate, ai)
    # This requires careful construction of option indices.
    # The original code: P_i = np.einsum('ijk,ijl->ijkl', Ps[-1], P).reshape((n_states, n_states, -1))
    # If Ps[-1] is (S, S', num_prev_options) and P is (S, S', num_actions) (transposed)
    # 'ijk' means (s, s_intermediate, prev_option_idx)
    # 'ijl' means (s, s_prime_final, current_action_idx) -> This P should be P(s_prime_final | s_intermediate, current_action)
    # So P should be indexed (s_intermediate, s_prime_final, current_action)
    # P_current_action = P_base_actions.transpose(1,2,0) # (A, S', S) ? No.
    # P_current_action should be (S_intermediate, S_prime_final, A_current)
    
    # Let P_prev_option = Ps_list[-1] be (S_start, S_end_prev_option, num_opts_len_prev)
    # Let P_action = P_base_actions.transpose(0,2,1) be (S_start_action, S_end_action, num_base_actions)
    # We want P_new_option(s_start, s_final, (opt_prev, action_curr))
    # = sum_{s_intermediate} P_prev_option(s_start, s_intermediate, opt_prev) * P_action(s_intermediate, s_final, action_curr)
    # einsum 'ikp, kjl -> ijpl' where p=num_opts_prev, l=num_actions. Then reshape (p*l) to new_num_opts.
    # i=s_start, k=s_intermediate, j=s_final
    
    for _ in range(num_steps_k - 1):
        P_prev_options: NDArray[jnp.float_] = Ps_list[-1] # (S, S', num_options_len_prev)
        # P_base_actions_T is (S,S',A)
        # P_len_i[s_start, s_final, (opt_prev_idx, current_action_idx)] =
        #   sum_{s_intermediate} P_prev_options[s_start, s_intermediate, opt_prev_idx] * P_base_actions_T[s_intermediate, s_final, current_action_idx]
        P_current_step_options: NDArray[jnp.float_] = jnp.einsum(
            'ikp,kjl->ijpl', 
            P_prev_options, 
            P_base_actions.transpose(0,2,1) # P(s_intermediate, s_final, action_current)
        ).reshape((num_states, num_states, -1)) # Reshape (S,S,P*L) to (S,S, num_new_options)
        Ps_list.append(P_current_step_options)
        
    return jnp.concatenate(Ps_list, axis=-1) # Concatenate along the options dimension


def lmdp_option_decoder(
    u_target_dynamics: NDArray[jnp.float_], 
    P_base_actions: NDArray[jnp.float_], 
    learning_rate: float = 1.0, 
    num_steps_k: int = 5,
    max_iterations: int = 1000, # Added for utils.solve
    tolerance: float = 1e-6     # Added for utils.solve
) -> NDArray[jnp.float_]:
    """
    Decodes target control dynamics `u_target_dynamics` into an option policy.
    This involves finding a policy over k-step options (sequences of base actions)
    such that the resulting dynamics under this option policy approximate `u_target_dynamics`.
    The option policy is parameterized by logits and learned using gradient descent to minimize KL divergence.

    Args:
        u_target_dynamics: Target optimal control dynamics, shape (S, S).
                           u_target_dynamics[s', s] = u*(s'|s) (column-stochastic).
                           Assumed u[s_row, s_col] means u(s_col | s_row).
        P_base_actions: Transition probability tensor for base actions, shape (S, A, S').
                        P_base_actions[s, a, s'] = P(s'|s,a).
        learning_rate: Learning rate for gradient descent.
        num_steps_k: The maximum length of action sequences (options) to consider.
        max_iterations: Max iterations for the solver.
        tolerance: Convergence tolerance for option logits.

    Returns:
        The learned option policy `pi_options` as a JAX array of shape (S, num_total_options),
        where `num_total_options` is sum_{i=1 to k} A^i.
        `pi_options[s, opt_idx]` is the probability of choosing option `opt_idx` in state `s`.
    """
    num_states: int = P_base_actions.shape[0]
    num_base_actions: int = P_base_actions.shape[1]

    # P_all_options[s, s', option_idx], where option_idx combines options of len 1 to k.
    # If P_all_options[s_row, s'_col, opt_idx] means P(s'_col | s_row, opt_idx)
    P_all_options: NDArray[jnp.float_] = option_transition_fn(P_base_actions, num_steps_k)
    num_total_options: int = P_all_options.shape[-1]
    # print(f'N options: {num_total_options}') # Original print

    def loss_fn(option_logits: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        # option_policy[s, opt_idx] = probability of choosing option opt_idx in state s
        option_policy: NDArray[jnp.float_] = utils.softmax(option_logits, axis=1) # Softmax over options for each state
        
        # P_under_option_policy[s, s'] = sum_{opt} P_all_options[s, s', opt] * option_policy[s, opt]
        # P_all_options is (S_start, S_end, num_options). P_all_options[s,s',opt] = P(s'|s,opt)
        # option_policy is (S_start, num_options). option_policy[s,opt] = pi(opt|s)
        # P_effective[s,s'] = sum_opt P(s'|s,opt)pi(opt|s)
        P_effective: NDArray[jnp.float_] = jnp.einsum('sok,so->sk', P_all_options, option_policy) # (S,S)
        
        # KL(u_target || P_effective)
        # Assuming u_target_dynamics[s,s'] = u(s'|s)
        kl_divergence: NDArray[jnp.float_] = jnp.sum(
            u_target_dynamics * jnp.log(u_target_dynamics / (P_effective + 1e-8) + 1e-8)
        )
        return kl_divergence

    dL_doption_logits: Callable = jit(grad(loss_fn))

    def update_fn_option_logits(current_option_logits: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        gradient: NDArray[jnp.float_] = dL_doption_logits(current_option_logits)
        return current_option_logits - learning_rate * gradient

    initial_option_logits: NDArray[jnp.float_] = rnd.standard_normal((num_states, num_total_options))
    
    final_logits_trajectory: List[NDArray[jnp.float_]] = utils.solve(
        update_fn_option_logits, 
        initial_option_logits,
        max_iter=max_iterations,
        tol=tolerance
    )
    optimal_option_logits: NDArray[jnp.float_] = final_logits_trajectory[-1]
    
    return utils.softmax(optimal_option_logits, axis=1)
