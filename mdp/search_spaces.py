"""
This module provides functions for creating and manipulating parameterized matrices,
and for defining update rules for various MDP algorithms and optimization routines.
The focus seems to be on exploring how different parameterizations and search
strategies affect the dynamics of learning and optimization.
"""
import functools
from typing import List, Callable, Tuple, Any, Sequence # Added Sequence

import numpy # For type hints if numpy specific features are used (e.g. linalg.pinv if not from jax)
import jax.numpy as jnp
from jax import grad, jit, jacrev, vmap
from numpy.typing import NDArray # For type hinting JAX arrays as well

import numpy.random as rnd # Standard numpy random for initializations

import mdp.utils as utils # Assuming this provides MDP class and other utilities

# --- Matrix Parameterization and Manipulation ---

def random_parameterised_matrix(
    input_dim: int, 
    output_dim: int, 
    hidden_dim: int, 
    num_hidden_layers: int
) -> List[NDArray[jnp.float_]]:
    """
    Creates a list of core matrices for a factorized representation of a larger matrix.
    This represents a simple feedforward neural network structure without non-linearities.
    Matrices are initialized using Glorot (Xavier) initialization.

    Args:
        input_dim: The input dimension of the overall matrix (e.g., number of states).
        output_dim: The output dimension of the overall matrix (e.g., number of actions).
        hidden_dim: The dimension of the hidden layers.
        num_hidden_layers: The number of hidden layers in the factorization.

    Returns:
        A list of JAX arrays (cores). The first matrix maps from input_dim to hidden_dim,
        intermediate matrices map from hidden_dim to hidden_dim, and the last matrix
        maps from hidden_dim to output_dim.
    """
    glorot_init = lambda shape: (1 / jnp.sqrt(shape[0] + shape[1])) * rnd.standard_normal(shape).astype(jnp.float32)
    
    cores: List[NDArray[jnp.float_]] = [glorot_init((hidden_dim, hidden_dim)) for _ in range(num_hidden_layers)]
    cores = [glorot_init((input_dim, hidden_dim))] + cores + [glorot_init((hidden_dim, output_dim))]
    return cores

def combine_svd(
    u_matrix: NDArray[jnp.float_], 
    s_vector: NDArray[jnp.float_], 
    vT_matrix: NDArray[jnp.float_]
) -> NDArray[jnp.float_]:
    """
    Reconstructs a matrix from its SVD components (U, S, V^T).

    Args:
        u_matrix: The U matrix from SVD.
        s_vector: The singular values (1D array S).
        vT_matrix: The V^T matrix from SVD.

    Returns:
        The reconstructed matrix: U @ diag(S) @ V^T.
    """
    return jnp.dot(u_matrix, jnp.dot(jnp.diag(s_vector), vT_matrix))

def random_reparameterisation(
    core_matrices: List[NDArray[jnp.float_]], 
    layer_index: int
) -> List[NDArray[jnp.float_]]:
    """
    Applies a random reparameterization to two adjacent core matrices in a list
    while preserving their product. This is done by manipulating their SVD components.
    This function's exact mathematical utility might be specific to research context.

    Args:
        core_matrices: A list of JAX arrays representing the factorized matrix.
        layer_index: The index of the first core matrix in the pair to be reparameterized.
                     `core_matrices[layer_index]` and `core_matrices[layer_index+1]` are used.
                     Must be `0 < layer_index < len(core_matrices) - 1`.

    Returns:
        A new list of core matrices with the specified pair reparameterized.
    """
    if not (0 < layer_index < len(core_matrices) - 1):
        raise ValueError("layer_index must be such that core_matrices[layer_index] and core_matrices[layer_index+1] exist and are not ends.")

    matrix_i: NDArray[jnp.float_] = core_matrices[layer_index]
    matrix_i_plus_1: NDArray[jnp.float_] = core_matrices[layer_index + 1]
    
    hidden_dim_n: int = matrix_i.shape[-1]
    hidden_dim_m: int = matrix_i_plus_1.shape[0]
    if hidden_dim_n != hidden_dim_m:
        raise ValueError("Inner dimensions of matrices for reparameterization must match.")

    u_i, s_i, vT_i = jnp.linalg.svd(matrix_i, full_matrices=False)
    u_ip1, s_ip1, vT_ip1 = jnp.linalg.svd(matrix_i_plus_1, full_matrices=False)
    
    # The reparameterization logic:
    # M1' = U1 * S1 * (V1^T * U2)
    # M2' = I * S2 * V2^T 
    # Product (M1'*M2') should be same as (M1*M2) if V1^T * U2 is unitary/orthogonal, or other specific conditions.
    # The original code seems to use identity for the U matrix of the second reconstructed part.
    
    new_matrix_i = combine_svd(u_i, s_i, jnp.dot(vT_i, u_ip1))
    # The `np.eye(n)` implies `n` should be `s_ip1.shape[0]`.
    # If s_ip1 is 1D vector of singular values, diag(s_ip1) is the Sigma matrix.
    # The U matrix for the second part is effectively `u_ip1` if we want to preserve M_i+1 structure,
    # but here it's forced to be identity: `combine_svd(jnp.eye(hidden_dim_n), s_ip1, vT_ip1)`
    # This means the second matrix becomes `diag(s_ip1) @ vT_ip1`.
    
    # Assuming hidden_dim_n is the dimension for the identity matrix.
    # This makes the second matrix `Sigma_{i+1} @ V^T_{i+1}` (missing U_{i+1}).
    new_matrix_i_plus_1 = combine_svd(jnp.eye(s_ip1.shape[0]), s_ip1, vT_ip1) 

    return core_matrices[:layer_index] + [new_matrix_i, new_matrix_i_plus_1] + core_matrices[layer_index + 2:]

def build(core_matrices: List[NDArray[jnp.float_]]) -> NDArray[jnp.float_]:
    """
    Reconstructs the full matrix by taking the dot product of a list of core matrices.

    Args:
        core_matrices: A list of JAX arrays (factorized matrices).

    Returns:
        The resulting full matrix as a JAX array.
    """
    if not core_matrices:
        raise ValueError("Core matrices list cannot be empty.")
    return functools.reduce(jnp.dot, core_matrices)

# --- MDP Algorithm Update Rule Generators ---

def sarsa(mdp_object: utils.MDP, learning_rate: float) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns the SARSA update function for Q-values.
    Q_next(s,a) = Q(s,a) + lr * (r + gamma * Q(s',a') - Q(s,a))
    where a' is chosen using the current policy (e.g., epsilon-greedy from Q).
    This simplified version uses a greedy policy pi(Q) to select a', then uses Q(s', pi(Q)(s')).

    Args:
        mdp_object: The MDP environment.
        learning_rate: The learning rate for the Q-value updates.

    Returns:
        A JIT-compiled callable update function `U(Q_current) -> Q_next`.
    """
    # Policy derived from Q (greedy for SARSA's next action value estimation part)
    # This pi is used to get Q(s', a') where a' = pi(s')
    pi_from_q = lambda Q_current: utils.onehot(jnp.argmax(Q_current, axis=1), Q_current.shape[1])
    
    # Bellman operator for SARSA: T(Q) = r + gamma * Q(s', pi(s'))
    # np.einsum('jk,jk->j', Q, pi(Q)) calculates sum_a Q(s,a)pi(a|s) for each state s.
    # This is E_{a~pi(s)}[Q(s,a)]. For SARSA, we need Q(s', a') where a' is from policy.
    # So, it should be V_pi(s') = Q(s', pi(s')).
    def T_sarsa(Q_current: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        policy_actions = pi_from_q(Q_current) # (S, A), one-hot actions for each state
        # Q_next_state_values = Q_current[s', a'] where a' = policy_actions(s')
        # This is effectively V_pi(s') based on current Q. Shape (S,).
        V_pi_s_prime: NDArray[jnp.float_] = jnp.einsum('sa,sa->s', Q_current, policy_actions)
        return utils.bellman_operator(mdp_object.P, mdp_object.r, V_pi_s_prime[:, jnp.newaxis], mdp_object.discount)

    update_rule = lambda Q_current: Q_current + learning_rate * (T_sarsa(Q_current) - Q_current)
    return jit(update_rule)

def q_learning(mdp_object: utils.MDP, learning_rate: float) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns the Q-learning update function for Q-values.
    Q_next(s,a) = Q(s,a) + lr * (r + gamma * max_a' Q(s',a') - Q(s,a))

    Args:
        mdp_object: The MDP environment.
        learning_rate: The learning rate for the Q-value updates.

    Returns:
        A JIT-compiled callable update function `U(Q_current) -> Q_next`.
    """
    # Bellman operator for Q-learning: T(Q) = r + gamma * max_a' Q(s',a')
    def T_q_learning(Q_current: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        V_max_s_prime: NDArray[jnp.float_] = jnp.max(Q_current, axis=1) # V(s') = max_a' Q(s',a'). Shape (S,)
        return utils.bellman_operator(mdp_object.P, mdp_object.r, V_max_s_prime[:, jnp.newaxis], mdp_object.discount)
        
    update_rule = lambda Q_current: Q_current + learning_rate * (T_q_learning(Q_current) - Q_current)
    return jit(update_rule)

def value_iteration(mdp_object: utils.MDP, learning_rate: float) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns the Value Iteration update function for state-values V.
    V_next(s) = V(s) + lr * (max_a (r(s,a) + gamma * sum_s' P(s'|s,a)V(s')) - V(s))
              = V(s) + lr * (max_a Q_V(s,a) - V(s))

    Args:
        mdp_object: The MDP environment.
        learning_rate: The learning rate (or step size, often 1 for exact VI).
                       If lr=1, it's V_next(s) = max_a Q_V(s,a).

    Returns:
        A JIT-compiled callable update function `U(V_current) -> V_next`.
    """
    def T_value_iteration(V_current: NDArray[jnp.float_]) -> NDArray[jnp.float_]: # Bellman optimality backup for V
        if V_current.shape[1] != 1:
             raise ValueError(f"V_current must be a column vector (S,1), got {V_current.shape}")
        # Q_for_V[s,a] = r(s,a) + gamma * sum_s' P(s'|s,a)V(s')
        Q_values_from_V: NDArray[jnp.float_] = utils.bellman_operator(mdp_object.P, mdp_object.r, V_current, mdp_object.discount)
        return jnp.max(Q_values_from_V, axis=1, keepdims=True) # max_a Q(s,a) -> V*(s)

    update_rule = lambda V_current: V_current + learning_rate * (T_value_iteration(V_current) - V_current)
    return jit(update_rule)

# thompson_abstraction_value_iteration requires sample_using_symmetric_prior which is not defined.
# Skipping detailed refactoring for it unless sample_using_symmetric_prior is provided/defined.
def thompson_abstraction_value_iteration(mdp_object: utils.MDP, learning_rate: float):
    """
    NOTE: This function depends on an undefined `sample_using_symmetric_prior`.
    Placeholder for Thompson sampling based value iteration with abstraction.
    """
    def T_vi(V_current: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        if V_current.shape[1] != 1:
            raise ValueError(f"V_current must be a column vector (S,1), got {V_current.shape}")
        return jnp.max(utils.bellman_operator(mdp_object.P, mdp_object.r, V_current, mdp_object.discount), axis=1, keepdims=True)

    @jit
    def update_fn(X: Tuple[NDArray[jnp.float_], Any]): # Any for S (similarity matrix)
        V_current, S_similarity_matrix = X
        # The following line will cause an error as sample_using_symmetric_prior is not defined.
        # X_sampled_abstraction = sample_using_symmetric_prior(S_similarity_matrix) 
        X_sampled_abstraction = jnp.eye(V_current.shape[0]) # Placeholder: no abstraction
        return V_current + learning_rate * jnp.dot(X_sampled_abstraction, (T_vi(V_current) - V_current))
    return update_fn


def parameterised_value_iteration(mdp_object: utils.MDP, learning_rate: float) -> Callable[[List[NDArray[jnp.float_]]], List[NDArray[jnp.float_]]]:
    """
    Returns an update function for parameterized Q-values (Q is `build(cores)`).
    Performs gradient ascent on Q-values w.r.t. core matrix parameters.
    The update is based on the Temporal Difference error of the Bellman optimality operator.

    Args:
        mdp_object: The MDP environment.
        learning_rate: Learning rate for updating core matrices.

    Returns:
        A JIT-compiled callable update function `U(cores_current) -> cores_next`.
    """
    # T_q_optimal is the Bellman optimality operator for Q-values
    T_q_optimal = lambda Q_values: utils.bellman_optimality_operator(mdp_object.P, mdp_object.r, Q_values, mdp_object.discount)
    
    # TD error: T_q_optimal(Q_theta) - Q_theta
    td_error_fn = lambda current_cores: T_q_optimal(build(current_cores)) - build(current_cores)
    
    # Jacobian of Q_theta (build(cores)) w.r.t. each core matrix
    # dQ_dcore_fn = jacrev(build) # jacrev returns a function that computes jacobians

    # For complex functions like `build` (reduce dot product), jacrev might be complex.
    # It's often easier to use `grad` of a scalar loss.
    # This implementation implies a specific structure for dVdw (should be dQ_dcores).
    # Let's assume dQ_dcores(cores) returns a list of Jacobians [dQ/dcore1, dQ/dcore2, ...]
    # where dQ/dcore_i has shape (S, A, core_i_shape_0, core_i_shape_1)
    
    # This part is tricky and requires careful JAX usage for Jacobians of matrix products.
    # The original einsum `delta (S,A) @ dQ/dcore (S,A,core_dims)` suggests specific grad structure.
    # For simplicity in refactoring without full re-derivation:
    
    # We need grad of a scalar loss. Let loss = sum( (T(Q_theta) - Q_theta)^2 ).
    # Or, use policy gradient theorem style: grad_cores = dQ_dcores * TD_error (element-wise then sum)
    # The original einsum 'ij,ijkl->kl' implies delta is (S,A) and dc (dQ/dcore) is (S,A,core_dim0,core_dim1)
    # and result is (core_dim0, core_dim1), which is the shape of the core.
    
    # This needs jacobian of build (S,A output) wrt each core in list of cores.
    # jacrev(build, argnums=0) will treat `cores` as a single arg (list).
    # It's better to define loss and take grad of loss.
    # However, following original structure:
    
    # Placeholder for dQ_dcore_fn if jacrev(build) is not directly usable this way.
    # This should be: d(build(cores))/d(core_i)
    # jacrev(build) will give a list of jacobians if build takes list of cores.
    # The list comprehension `[np.einsum('ij,ijkl->kl', delta, dc) for dc in dVdw(cores)]`
    # implies dVdw (should be dQdw) returns a list of Jacobians, one for each core.
    
    # JAX's jacrev is powerful. If `build` is a JAX-traceable function of `cores` (list of JAX arrays),
    # `jacrev(build)` will return a list of Jacobians, one for each core matrix in `cores`.
    # Each Jacobian `dc_i` will have shape `(output_shape_of_build) + core_i.shape`.
    # If build(cores) outputs Q (S,A), then dc_i is (S, A, core_i_rows, core_i_cols).
    
    dQ_dcores_fn = jacrev(build)


    @jit
    def update_rule_cores(current_cores: List[NDArray[jnp.float_]]) -> List[NDArray[jnp.float_]]:
        td_error: NDArray[jnp.float_] = td_error_fn(current_cores) # Shape (S, A)
        
        # List of Jacobians: dQ/d_core_i, each shaped (S, A, core_i_rows, core_i_cols)
        core_jacobians: List[NDArray[jnp.float_]] = dQ_dcores_fn(current_cores) 
        
        # Compute gradient for each core: sum_{s,a} TD_error[s,a] * dQ[s,a]/d_core_i
        # Original einsum: 'ij,ijkl->kl' where ij is (S,A) and ijkl is (S,A,core_rows,core_cols)
        # This effectively sums over S and A.
        core_gradients: List[NDArray[jnp.float_]] = [
            jnp.einsum('sa,sakl->kl', td_error, core_jac) for core_jac in core_jacobians
        ]
        
        updated_cores: List[NDArray[jnp.float_]] = [
            core + learning_rate * grad for core, grad in zip(current_cores, core_gradients)
        ]
        return updated_cores
    return update_rule_cores # Original did jit(update_fn), JIT applies to the returned function.

def complex_value_iteration(mdp_object: utils.MDP, learning_rate: float) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns a Value Iteration-like update rule using a squared error loss on absolute Q-values.
    The purpose of `abs(Q)` here is unclear without more context, might be specific research.
    L(Q) = sum ( (T(abs(Q)) - abs(Q))^2 )
    Update: Q_next = Q - lr * dL/dQ

    Args:
        mdp_object: The MDP environment.
        learning_rate: Learning rate for the Q-value updates.

    Returns:
        A JIT-compiled callable update function `U(Q_current) -> Q_next`.
    """
    T_q_optimal = lambda Q_values: utils.bellman_optimality_operator(mdp_object.P, mdp_object.r, Q_values, mdp_object.discount)
    
    # Loss based on absolute Q-values
    loss_fn = lambda Q_current: jnp.sum((T_q_optimal(jnp.abs(Q_current)) - jnp.abs(Q_current))**2)
    
    dLoss_dQ = grad(loss_fn)
    
    update_rule = lambda Q_current: Q_current - learning_rate * dLoss_dQ(Q_current)
    return jit(update_rule)

# --- Policy Iteration and Gradients ---

def policy_iteration(mdp_object: utils.MDP) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns the Policy Iteration update function.
    1. Policy Evaluation: V_pi = (I - gamma * P_pi)^-1 * r_pi
    2. Policy Improvement: pi_next(s) = argmax_a Q_pi(s,a), where Q_pi(s,a) = r(s,a) + gamma * sum_s' P(s'|s,a)V_pi(s')

    Args:
        mdp_object: The MDP environment.

    Returns:
        A callable update function `U(policy_current) -> policy_next`.
    """
    def update_rule_policy(current_policy: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        # Policy Evaluation
        V_current_policy: NDArray[jnp.float_] = utils.value_functional(
            mdp_object.P, mdp_object.r, current_policy, mdp_object.discount
        )
        # Policy Improvement: Q-values for V_current_policy
        Q_for_improvement: NDArray[jnp.float_] = utils.bellman_operator(
            mdp_object.P, mdp_object.r, V_current_policy, mdp_object.discount
        )
        # New greedy policy (one-hot encoded)
        next_policy: NDArray[jnp.float_] = utils.onehot(jnp.argmax(Q_for_improvement, axis=1), mdp_object.A)
        return next_policy
    return update_rule_policy # JIT can be added if beneficial, but involves matrix inversion.

def policy_gradient_iteration_logits(mdp_object: utils.MDP, learning_rate: float, entropy_regularization: float = 1e-8) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns a policy gradient update function for policy logits.
    Uses REINFORCE-like gradient: grad_logits V_pi + entropy_bonus.
    V_pi = sum_s d0(s)V_pi(s) (or average V_pi(s) over states). Here, using sum V_pi(s).

    Args:
        mdp_object: The MDP environment.
        learning_rate: Learning rate for updating policy logits.
        entropy_regularization: Coefficient for entropy bonus to encourage exploration.

    Returns:
        A JIT-compiled callable update function `U(logits_current) -> logits_next`.
    """
    # Gradient of entropy H(softmax(logits)) w.r.t. logits
    dEntropy_dlogits_fn = grad(lambda logits: utils.entropy(utils.softmax(logits, axis=1)))
    
    # Gradient of sum V_pi(s) w.r.t. logits
    # V_pi(s) is value of policy softmax(logits)
    dSumV_dlogits_fn = grad(lambda logits: jnp.sum(
        utils.value_functional(mdp_object.P, mdp_object.r, utils.softmax(logits, axis=1), mdp_object.discount)
    ))

    @jit
    def update_rule_logits(current_logits: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        # Policy gradient for sum V_pi(s)
        grad_sum_V: NDArray[jnp.float_] = dSumV_dlogits_fn(current_logits)
        
        # Entropy gradient for regularization
        grad_entropy: NDArray[jnp.float_] = dEntropy_dlogits_fn(current_logits)
        
        # Clip norm of policy gradient (optional, from original code)
        clipped_grad_sum_V: NDArray[jnp.float_] = utils.clip_by_norm(grad_sum_V, max_norm=500.0) # Max norm from original
        
        next_logits: NDArray[jnp.float_] = current_logits + learning_rate * clipped_grad_sum_V + entropy_regularization * grad_entropy
        return next_logits
    return update_rule_logits


def parameterised_policy_gradient_iteration(mdp_object: utils.MDP, learning_rate: float, entropy_reg_coeff: float = 1e-6) -> Callable[[List[NDArray[jnp.float_]]], List[NDArray[jnp.float_]]]:
    """
    Returns an update function for parameters (cores) of a factorized policy representation.
    Policy pi_cores = softmax(build(cores)). Uses Actor-Critic style update.
    grad_cores = sum_{s,a} (Q_pi(s,a) - V_pi(s)) * d(log pi_cores(a|s))/d(core_i) + entropy_bonus

    Args:
        mdp_object: The MDP environment.
        learning_rate: Learning rate for updating core matrices.
        entropy_reg_coeff: Coefficient for entropy regularization.

    Returns:
        A JIT-compiled callable update function `U(cores_current) -> cores_next`.
    """
    
    def get_policy_and_values(current_cores: List[NDArray[jnp.float_]]) -> Tuple[NDArray[jnp.float_], NDArray[jnp.float_], NDArray[jnp.float_]]:
        current_policy: NDArray[jnp.float_] = utils.softmax(build(current_cores), axis=1)
        V_pi: NDArray[jnp.float_] = utils.value_functional(mdp_object.P, mdp_object.r, current_policy, mdp_object.discount)
        Q_pi: NDArray[jnp.float_] = utils.bellman_operator(mdp_object.P, mdp_object.r, V_pi, mdp_object.discount)
        return current_policy, V_pi, Q_pi

    # We need jacobian of log pi(.|s) w.r.t each core_i.
    # log_pi_fn = lambda crs: jnp.log(utils.softmax(build(crs), axis=1) + 1e-8)
    # dlogpi_dcores_fn = jacrev(log_pi_fn) # List of Jacobians [dlogpi/dcore1, ...]
                                        # Each jac_i shape: (S,A) + core_i.shape
    
    # And jacobian of entropy H(pi) w.r.t. each core_i
    # entropy_fn = lambda crs: utils.entropy(utils.softmax(build(crs), axis=1))
    # dEntropy_dcores_fn = jacrev(entropy_fn) # List of Jacobians [dH/dcore1, ...]
                                           # Each jac_i shape: core_i.shape (since H is scalar)

    # Using grad of a scalar loss is often more stable for complex parameterizations.
    # Loss_actor = - sum (Advantage * log pi). Loss_critic = MSE(V_target, V_current).
    # Here, it seems to be a direct policy gradient update.
    # The original einsum 'ijkl,ij->kl' implies dlogpi_dw (Jacobian of logpi w.r.t core)
    # has shape (S,A) + core_shape, and Advantage A is (S,A).

    # Re-deriving gradients using JAX's grad for a scalar objective (expected total advantage-weighted log-prob)
    # Objective_actor(cores) = sum_{s,a} Advantage(s,a) * log pi_cores(a|s)
    # Objective_entropy(cores) = Entropy(pi_cores)

    def actor_objective_fn(current_cores: List[NDArray[jnp.float_]], 
                           current_policy: NDArray[jnp.float_], # Detached, from previous calculation
                           advantages: NDArray[jnp.float_] # Detached (Q-V)
                          ) -> NDArray[jnp.float_]:
        # policy_from_cores = utils.softmax(build(current_cores), axis=1) # This is what we differentiate
        log_policy_from_cores = jnp.log(utils.softmax(build(current_cores), axis=1) + 1e-8)
        return jnp.sum(advantages * log_policy_from_cores)

    dActorObjective_dcores_fn = grad(actor_objective_fn, argnums=0) # Grad w.r.t. current_cores

    def entropy_objective_fn(current_cores: List[NDArray[jnp.float_]]) -> NDArray[jnp.float_]:
        return utils.entropy(utils.softmax(build(current_cores), axis=1))
    
    dEntropyObjective_dcores_fn = grad(entropy_objective_fn, argnums=0)


    @jit
    def update_rule_cores_actor_critic(current_cores: List[NDArray[jnp.float_]]) -> List[NDArray[jnp.float_]]:
        current_policy, V_pi, Q_pi = get_policy_and_values(current_cores)
        advantages: NDArray[jnp.float_] = Q_pi - V_pi # Shape (S,A)

        # Gradients for actor objective
        actor_grads: List[NDArray[jnp.float_]] = dActorObjective_dcores_fn(current_cores, current_policy, advantages)
        
        # Gradients for entropy objective
        entropy_grads: List[NDArray[jnp.float_]] = dEntropyObjective_dcores_fn(current_cores)
        
        updated_cores: List[NDArray[jnp.float_]] = [
            core + learning_rate * utils.clip_by_norm(actor_grad, max_norm=100.0) + entropy_reg_coeff * entropy_grad
            for core, actor_grad, entropy_grad in zip(current_cores, actor_grads, entropy_grads)
        ]
        return updated_cores
        
    return update_rule_cores_actor_critic


# --- Model Iteration ---

def parse_model_params(
    num_states: int, 
    num_actions: int, 
    flat_params: NDArray[jnp.float_]
) -> Tuple[NDArray[jnp.float_], NDArray[jnp.float_]]:
    """
    Reshapes a flat vector of parameters into transition (P_logits) and reward (r) components.

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A).
        flat_params: A 1D JAX array containing flattened parameters for P_logits and r.
                     Total length should be S*S*A (for P_logits) + S*A (for r).

    Returns:
        A tuple (P_logits, r_model):
            - P_logits: Logits for transition probabilities, shape (S_next, S_current, A).
            - r_model: Reward matrix, shape (S_current, A).
    """
    # P_logits part: S_next * S_current * A
    # P_tensor in MDP is (S_next, S_current, A)
    # r_matrix in MDP is (S_current, A)
    
    len_P_logits: int = num_states * num_states * num_actions
    P_logits_flat: NDArray[jnp.float_] = flat_params[:len_P_logits]
    r_model_flat: NDArray[jnp.float_] = flat_params[len_P_logits:]
    
    P_logits: NDArray[jnp.float_] = P_logits_flat.reshape((num_states, num_states, num_actions)) # (S_next, S_current, A)
    r_model: NDArray[jnp.float_] = r_model_flat.reshape((num_states, num_actions)) # (S_current, A)
    
    return P_logits, r_model

def model_iteration(
    true_mdp_object: utils.MDP, 
    learning_rate: float, 
    evaluation_policies: List[NDArray[jnp.float_]]
) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns an update function for model parameters (P_logits, r_model) based on minimizing
    the difference between value functions under the true MDP and the learned model,
    evaluated over a set of fixed policies.

    Args:
        true_mdp_object: The true MDP environment.
        learning_rate: Learning rate for updating model parameters.
        evaluation_policies: A list of policies to use for evaluating model accuracy.

    Returns:
        A JIT-compiled callable update function `U(params_current) -> params_next`.
    """
    # Pre-compute true value functions for evaluation policies under the true MDP
    # Vmap for batching over policies
    V_true_fn_batch = vmap(
        lambda policy: utils.value_functional(true_mdp_object.P, true_mdp_object.r, policy, true_mdp_object.discount)
    )
    V_true_values_batched: NDArray[jnp.float_] = V_true_fn_batch(jnp.stack(evaluation_policies)) # (Num_policies, S, 1)

    # Value function under the learned model (P_model, r_model)
    # P_model is softmax(P_logits) over S_next
    V_model_fn_batch = vmap(
        lambda P_model, r_model, policy: utils.value_functional(P_model, r_model, policy, true_mdp_object.discount), 
        in_axes=(None, None, 0) # P_model, r_model are fixed for this batch; policy varies
    )

    def loss_fn(model_params_flat: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        P_logits_model, r_model = parse_model_params(true_mdp_object.S, true_mdp_object.A, model_params_flat)
        # Normalize P_logits to get P_model (stochastic over S_next)
        P_model: NDArray[jnp.float_] = utils.softmax(P_logits_model, axis=0) # Axis 0 is S_next
        
        V_model_values_batched: NDArray[jnp.float_] = V_model_fn_batch(P_model, r_model, jnp.stack(evaluation_policies))
        
        # Sum of squared errors between true V and model V, over policies and states
        loss_value: NDArray[jnp.float_] = jnp.sum((V_true_values_batched - V_model_values_batched)**2)
        return loss_value

    dLoss_dparams_fn = grad(loss_fn)

    @jit
    def update_rule_model_params(current_params_flat: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        gradient: NDArray[jnp.float_] = dLoss_dparams_fn(current_params_flat)
        # Clip norm of gradient (optional, from original code)
        clipped_gradient: NDArray[jnp.float_] = utils.clip_by_norm(gradient, max_norm=100.0)
        return current_params_flat - learning_rate * clipped_gradient

    return update_rule_model_params

# --- Optimizer Utilities (Bundlers) ---

def momentum_bundler(
    base_update_fn: Callable[[Any], Any], # Takes W_t, returns W_{t+1} (or W_t + step)
    momentum_decay_rate: float # beta1 or decay coefficient
) -> Callable[[Tuple[Any, Any]], Tuple[Any, Any]]:
    """
    Wraps a base update function to include momentum.
    The state becomes `(parameters, momentum_vector)`.
    Update rule:
        gradient_estimate = base_update_fn(params) - params  (if base_update_fn returns new_params)
                            OR base_update_fn(params) (if base_update_fn returns step = lr*grad)
        momentum_next = decay * momentum_current + gradient_estimate
        params_next = params + (1 - decay) * momentum_next (This is non-standard momentum application)
        Standard momentum: params_next = params + momentum_next OR params + lr * momentum_next

    Args:
        base_update_fn: A callable that takes current parameters and returns updated parameters
                        (e.g., `param - lr * grad`). The difference `update_fn(W_t) - W_t` is used
                        as an estimate of `-lr * gradient`.
        momentum_decay_rate: The momentum decay factor (often called beta1 or mu).

    Returns:
        A JIT-compiled callable update function that takes `(params, momentum)`
        and returns `(params_next, momentum_next)`.
    """
    def momentum_update_rule(state: Tuple[Any, Any]) -> Tuple[Any, Any]:
        current_params, current_momentum = state[0], state[1]

        # Estimate step (e.g., -lr * gradient) from the base_update_fn
        # dW = update_fn(W_t) - W_t implies update_fn(W_t) is W_t - lr*g, so dW = -lr*g
        # This interpretation might be specific. If base_update_fn returns the gradient directly,
        # this needs adjustment. Assuming it returns W_t + step.
        
        if isinstance(current_params, jnp.ndarray):
            step_estimate: NDArray[jnp.float_] = base_update_fn(current_params) - current_params 
            next_momentum: NDArray[jnp.float_] = momentum_decay_rate * current_momentum + step_estimate
            # Original application: W_tp1 = W_t + (1 - decay) * M_tp1. This is unusual.
            # Standard momentum: W_tp1 = W_t + M_tp1 (if M_tp1 already includes learning rate)
            # Or: W_tp1 = W_t + learning_rate * M_tp1 (if M_tp1 is just momentum-averaged gradient)
            # Given dW is -lr*g, M_tp1 becomes average of -lr*g.
            # W_tp1 = W_t + M_tp1 (if 1-decay factor is absorbed in lr of base_update_fn)
            # For now, matching original:
            next_params: NDArray[jnp.float_] = current_params + (1 - momentum_decay_rate) * next_momentum
        elif isinstance(current_params, list): # For list of core matrices
            updated_params_from_base: List[NDArray[jnp.float_]] = base_update_fn(current_params)
            step_estimates: List[NDArray[jnp.float_]] = [
                updated_core - current_core 
                for updated_core, current_core in zip(updated_params_from_base, current_params)
            ]
            next_momentums: List[NDArray[jnp.float_]] = [
                momentum_decay_rate * mom_core + step_core 
                for mom_core, step_core in zip(current_momentum, step_estimates)
            ]
            next_params: List[NDArray[jnp.float_]] = [
                current_core + (1 - momentum_decay_rate) * mom_core 
                for current_core, mom_core in zip(current_params, next_momentums)
            ]
        else:
            raise ValueError(f"Unsupported parameter type for momentum_bundler: {type(current_params)}")

        return next_params, next_momentums
    return jit(momentum_update_rule)


def approximate(
    target_vector: NDArray[jnp.float_], 
    core_matrices: List[NDArray[jnp.float_]], 
    learning_rate: float = 1e-2, 
    activation_fn: Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]] = lambda x: x,
    max_iterations: int = 1000, # Added for utils.solve
    tolerance: float = 1e-7     # Added for utils.solve
) -> List[NDArray[jnp.float_]]:
    """
    Approximates a target vector `v` by optimizing the `core_matrices` that form `build(cores)`.
    Minimizes || activation_fn(build(cores)) - target_vector ||_2^2 using gradient descent with momentum.

    Example usage from original comment:
    cores = random_parameterised_matrix(2, 1, d_hidden=8, n_hidden=4)
    v = rnd.standard_normal((2,1))
    cores_ = approximate(v, cores)
    print(v, '\n',build(cores_))

    Args:
        target_vector: The target vector to approximate.
        core_matrices: Initial list of core matrices.
        learning_rate: Learning rate for the optimization.
        activation_fn: Activation function applied to `build(cores)` before comparing to target.
        max_iterations: Max iterations for the solver.
        tolerance: Convergence tolerance for the solver.

    Returns:
        The list of optimized core matrices.
    """
    def loss_fn(current_cores: List[NDArray[jnp.float_]]) -> NDArray[jnp.float_]:
        reconstruction: NDArray[jnp.float_] = activation_fn(build(current_cores))
        return jnp.sum(jnp.square(target_vector - reconstruction))

    dLoss_dcores_fn = grad(loss_fn) # Returns a list of gradients, one for each core

    # Base update rule (gradient descent step for cores)
    def l2_gradient_update_fn(current_cores: List[NDArray[jnp.float_]]) -> List[NDArray[jnp.float_]]:
        core_gradients: List[NDArray[jnp.float_]] = dLoss_dcores_fn(current_cores)
        updated_cores: List[NDArray[jnp.float_]] = [
            core - learning_rate * grad_core
            for core, grad_core in zip(current_cores, core_gradients)
        ]
        return updated_cores

    # Initial state for momentum solver: (initial_cores, zero_momentums)
    initial_momentums: List[NDArray[jnp.float_]] = [jnp.zeros_like(core) for core in core_matrices]
    initial_solver_state: Tuple[List[NDArray[jnp.float_]], List[NDArray[jnp.float_]]] = (core_matrices, initial_momentums)
    
    # Solve using momentum
    # Original used momentum_decay_rate = 0.9
    final_state_trajectory: List[Tuple[List[NDArray[jnp.float_]], List[NDArray[jnp.float_]]]] = utils.solve(
        momentum_bundler(l2_gradient_update_fn, momentum_decay_rate=0.9), 
        initial_solver_state,
        max_iter=max_iterations,
        tol=tolerance
    )
    
    optimized_cores, _ = final_state_trajectory[-1] # Get the parameter part of the last state
    return optimized_cores

# Note: The `gradient_bundler` function was mentioned in the prompt but is not in the original code.
# If it were to be implemented, it would likely be a simpler version of `momentum_bundler`
# that just applies `params - lr * grad(loss_fn)(params)`.

# The __main__ block from the original file is typically moved to tests or examples.
# It demonstrated usage of random_parameterised_matrix, build, and had commented out
# sections for policy_gradient_iteration and clip_by_norm tests.
