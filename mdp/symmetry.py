"""
This module focuses on constructing and exploring MDPs with specific structural
properties, such as those generated from a latent space or exhibiting certain symmetries.
"""
import functools # Not used in current refactored code, but was in original imports.
from typing import List, Tuple, Any, Callable # Added Any, Callable

import numpy # For type hints if numpy specific features are used
import jax.numpy as jnp
from jax import grad, jit, jacrev, vmap # jacrev not used in current refactored code
from numpy.typing import NDArray # For type hinting JAX arrays as well

import numpy.random as rnd # Standard numpy random for initializations

import mdp.utils as utils
import mdp.search_spaces as ss # Assuming this is search_spaces.py from this project

def generate_latent_mdp(
    num_states: int, 
    num_actions: int, 
    num_latent_dims: int, # Renamed from n_hidden
    default_discount: float = 0.5 # Added a default discount
) -> utils.MDP:
    """
    Generates an MDP where transition probabilities are formed via a low-rank factorization,
    implying a latent space structure. P(s'|s,a) = U_a @ V_a^T.
    The U_a and V_a matrices are generated randomly for each action.

    Args:
        num_states: Number of observable states (S).
        num_actions: Number of actions (A).
        num_latent_dims: Dimensionality of the latent space used for factorization (d_hidden).
        default_discount: Default discount factor to use for the generated MDP.

    Returns:
        An `mdp.utils.MDP` namedtuple instance with the generated parameters.
        P tensor shape will be (S_next, S_current, A) due to how P_action is constructed
        and then stacked. If P_action is P(s'|s) for a fixed action,
        then P_action[s',s]. Stacking along axis=-1 makes it P[s',s,a].
        This matches the P convention in `utils.MDP`.
    """
    # TODO: (from original) Design a structured state space where state partitions are guessable.
    
    P_actions_list: List[NDArray[jnp.float_]] = []
    for _ in range(num_actions):
        # For each action, P_a(s', s) = U_matrix @ VT_matrix
        # U_matrix maps current states to latent space, VT_matrix maps latent space to next states.
        # Or U_matrix (S, d_hidden), VT_matrix (d_hidden, S) -> P_a (S,S)
        # U_matrix[s, k], VT_matrix[k, s']
        
        # Original: U = rnd.random((num_states, num_latent_dims)) -> U[s, k_latent]
        #           U = U / np.sum(U, axis=0) (normalize columns of U)
        #           VT = rnd.random((num_latent_dims, num_states)) -> VT[k_latent, s']
        #           VT = VT / np.sum(VT, axis=0) (normalize columns of VT, which are s')
        #           P_action = np.dot(U, VT) -> P_action[s, s']
        # This makes P_action[s,s'] a transition matrix P(s'|s) for a fixed action.
        # So, P_action needs to be column-stochastic (sum over s' = 1).
        # The normalization in original code was over axis=0 for U and VT.
        # sum(U,axis=0) for U(S, H) normalizes each of H columns.
        # sum(VT,axis=0) for VT(H,S) normalizes each of S columns.
        # If P_a = U @ VT, then sum_{s'} P_a[s,s'] = sum_{s'} sum_k U[s,k]VT[k,s']
        #                                        = sum_k U[s,k] * (sum_{s'} VT[k,s'])
        # For P_a to be row-stochastic (sum_{s'} P_a[s,s'] = 1), we need sum_k U[s,k]*(sum_{s'} VT[k,s']) = 1.
        # The original normalization doesn't guarantee this.
        # Let's generate P_a and then normalize its rows.
        
        U_matrix_np: NDArray[numpy.float_] = rnd.random((num_states, num_latent_dims))
        VT_matrix_np: NDArray[numpy.float_] = rnd.random((num_latent_dims, num_states))
        
        P_action_np: NDArray[numpy.float_] = numpy.dot(U_matrix_np, VT_matrix_np) # P_action_np[s, s']
        
        # Normalize rows of P_action_np to make it P(s'|s, a)
        row_sums = P_action_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 # Avoid division by zero; effectively P(s'|s,a)=0 if all entries were 0.
        P_action_normalized_np: NDArray[numpy.float_] = P_action_np / row_sums
        
        P_actions_list.append(jnp.array(P_action_normalized_np))

    # Stack along the action dimension.
    # If each P_action_normalized_np is (S_current, S_next),
    # stacking them along axis=-1 gives (S_current, S_next, A).
    # MDP.P expects (S_next, S_current, A). So, transpose each P_action or the final stack.
    # Each P_action_normalized_np is P(s'|s,a_fixed). So P[s,s']. Transpose to P[s',s] for stacking.
    P_tensor_np: NDArray[numpy.float_] = numpy.stack([p.T for p in P_actions_list], axis=-1) # Now (S_next, S_current, A)

    # NOTE: (from original) The reward function should also ideally share this latent structure. How?
    r_matrix_np: NDArray[numpy.float_] = rnd.standard_normal((num_states, num_actions)) # r[s_current, action]
    d0_np: NDArray[numpy.float_] = rnd.random((num_states, 1))
    d0_normalized_np: NDArray[numpy.float_] = d0_np / d0_np.sum()

    return utils.MDP(
        S=num_states, 
        A=num_actions, 
        P=jnp.array(P_tensor_np), 
        r=jnp.array(r_matrix_np), 
        discount=default_discount, 
        d0=jnp.array(d0_normalized_np)
    )

def find_symmetric_mdp(
    num_states: int, 
    num_actions: int, 
    discount_factor: float, 
    learning_rate: float = 1e-2,
    max_iterations: int = 100, # Added for solver
    tolerance: float = 1e-5    # Added for solver
) -> utils.MDP:
    """
    Attempts to find an MDP that exhibits policy symmetry by optimizing model parameters.
    Policy symmetry here means V(pi_flipped) is close to V(pi) for all deterministic policies `pi`,
    where `pi_flipped` is the policy with action indices flipped (e.g., if A=2, action 0 becomes 1, 1 becomes 0).
    The optimization minimizes the squared difference between V(pi) and V(pi_flipped).

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A). (Flipping assumes A is known, e.g. 2 for simple flip).
        discount_factor: Discount factor (gamma).
        learning_rate: Learning rate for optimizing model parameters.
        max_iterations: Max iterations for the solver.
        tolerance: Convergence tolerance for model parameters.


    Returns:
        An `mdp.utils.MDP` instance with parameters (P, r) optimized for policy symmetry.
        P tensor will be (S_next, S_current, A).
    """
    # Initial guess for model parameters (P_logits and r flattened)
    # P_logits are (S_next, S_current, A), r is (S_current, A)
    initial_model_params_flat_np: NDArray[numpy.float_] = rnd.standard_normal(
        num_states * num_states * num_actions + num_states * num_actions
    )
    
    deterministic_policies_list: List[NDArray[jnp.float_]] = utils.get_deterministic_policies(num_states, num_actions)
    # Stack into a JAX array: (Num_det_policies, S, A)
    det_policies_tensor: NDArray[jnp.float_] = jnp.stack(deterministic_policies_list)

    # Vmap for batching value function calculation over policies
    # Assumes value_functional takes P(S_next,S_curr,A), r(S_curr,A), pi(S_curr,A)
    # The mdp_object argument is missing in the original V and V_guess vmap calls.
    # It should be: vmap(lambda P_model, r_model, policy: utils.value_functional(P_model, r_model, policy, discount_factor), ...)
    # However, the loss_fn below calls V(utils.softmax(P), r, pis) which implies V itself is the vmapped function
    # that takes P and r as args. This is unusual.
    # Let's define V_batch_fn correctly.
    
    # V_batch_fn: (P_model, r_model, policies_tensor) -> V_values_batched
    # policies_tensor is (Num_policies, S, A)
    # P_model (S_next,S_curr,A), r_model (S_curr,A)
    # utils.value_functional returns (S_curr, 1)
    # So V_values_batched should be (Num_policies, S_curr, 1)
    V_batch_fn = vmap(
        lambda P_model, r_model, policy: utils.value_functional(P_model, r_model, policy, discount_factor), 
        in_axes=(None, None, 0) # P_model, r_model are fixed for this batch; policy varies
    )


    def loss_fn(model_params_flat: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        # Parse flat params into P_logits (S_next,S_curr,A) and r_model (S_curr,A)
        P_logits_model, r_model = ss.parse_model_params(num_states, num_actions, model_params_flat)
        # Normalize P_logits to get P_model (stochastic over S_next for each s_curr, a)
        P_model: NDArray[jnp.float_] = utils.softmax(P_logits_model, axis=0) 
        
        # Value functions for original deterministic policies
        # The original code `V(utils.softmax(P), r, pis)` is problematic because V was not defined
        # to take P and r. Assuming V_batch_fn is intended.
        V_original_policies: NDArray[jnp.float_] = V_batch_fn(P_model, r_model, det_policies_tensor) # (Num_policies, S, 1)
        
        # Flipped policies (e.g., action 0 maps to A-1, 1 to A-2, etc.)
        # This assumes a specific way to "flip". jnp.flip(axis=1) flips the one-hot encoding
        # if actions are the last dimension of policies. If A=2, [1,0] becomes [0,1].
        # If policies are (Num, S, A), flip along action dim (axis=2 or -1).
        # Original: np.flip(pis, 1) -> if pis is (Num, S, A), axis 1 is S. This flips state order for actions.
        # This is not action flipping.
        # To flip actions, e.g., for A=2, map action 0 to 1, 1 to 0.
        # If policies are one-hot (N,S,A), then jnp.flip(policies, axis=2) would flip action indices if A is ordered.
        # For a general flip if A > 2, a permutation matrix or explicit re-indexing is needed.
        # Assuming simple flip for A=2, or general reversal of action indices.
        # For one-hot, flipping the one-hot vector is `policy_flipped[s, a] = policy[s, A-1-a]`.
        # jnp.flip(det_policies_tensor, axis=2) achieves this if action dim is last.
        if num_actions > 1 : # Ensure there's something to flip
            flipped_policies_tensor: NDArray[jnp.float_] = jnp.flip(det_policies_tensor, axis=2) # Flips action dimension
        else: # No change if only one action
            flipped_policies_tensor = det_policies_tensor

        V_flipped_policies: NDArray[jnp.float_] = V_batch_fn(P_model, r_model, flipped_policies_tensor)
        
        # Loss is sum of squared differences between V(pi) and V(pi_flipped)
        loss_value: NDArray[jnp.float_] = jnp.sum(jnp.square(V_original_policies - V_flipped_policies))
        return loss_value

    # def loss_fn_value_symmetry(model_params_flat: NDArray[jnp.float_]): # Original commented out alternative
    #     P_logits_model, r_model = ss.parse_model_params(num_states, num_actions, model_params_flat)
    #     P_model = utils.softmax(P_logits_model, axis=0)
    #     vals = V_batch_fn(P_model, r_model, det_policies_tensor) # (Num_policies, S, 1)
    #     num_half_states = num_states // 2
    #     # Compares V(s_i) with V(s_{i+n/2})
    #     return jnp.sum(jnp.square(vals[:, :num_half_states, 0] - vals[:, num_half_states:2*num_half_states, 0]))


    dLoss_dparams_fn = grad(loss_fn)
    
    # Renamed `update_fn` to `update_rule_model_params` for clarity
    def update_rule_model_params(current_params_flat: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        gradient: NDArray[jnp.float_] = dLoss_dparams_fn(current_params_flat)
        return current_params_flat - learning_rate * gradient # Basic gradient descent

    # Initial state for solver (params, momentum_vector if using momentum_bundler)
    # Original used momentum_bundler.
    initial_solver_state: Tuple[NDArray[jnp.float_], NDArray[jnp.float_]] = (
        jnp.array(initial_model_params_flat_np), 
        jnp.zeros_like(initial_model_params_flat_np)
    )
    
    final_model_params_flat, _ = utils.solve( # _ for momentum_var
        ss.momentum_bundler(update_rule_model_params, momentum_decay_rate=0.9), # Original decay=0.9
        initial_solver_state,
        max_iter=max_iterations,
        tol=tolerance
    )[-1]

    P_logits_optimized, r_optimized = ss.parse_model_params(num_states, num_actions, final_model_params_flat)
    P_optimized: NDArray[jnp.float_] = utils.softmax(P_logits_optimized, axis=0) # Normalize over S_next
    
    d0_np: NDArray[numpy.float_] = rnd.random((num_states, 1))
    d0_normalized: NDArray[jnp.float_] = jnp.array(d0_np / d0_np.sum())
    
    return utils.MDP(
        S=num_states, 
        A=num_actions, 
        P=P_optimized, 
        r=r_optimized, 
        discount=discount_factor, 
        d0=d0_normalized
    )


def sample_using_symmetric_prior(S_similarity_matrix: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
    """
    TODO: This function is a placeholder and not fully implemented.
    Its intent seems to be sampling a matrix X (e.g., an abstraction or grouping matrix)
    based on a similarity matrix S, potentially using automorphisms (autG) of S.

    Args:
        S_similarity_matrix: A similarity matrix where S[i,j]=1 indicates states i and j are similar.

    Returns:
        A matrix X, likely related to state groupings or an abstraction, based on S.
        The original comment suggests X[i,j]=1 if S_ij ~= 1, and X is in [0,1]^(nS x nS).
    """
    # Original code:
    # autG = automorphisms(S_similarity_matrix) # `automorphisms` is not defined in this file or common libraries.
    # return X # X is not defined.
    
    # This function needs a proper definition of `automorphisms` and how X is derived.
    # For now, returning a placeholder or raising NotImplementedError.
    print("Warning: sample_using_symmetric_prior is not implemented and requires an 'automorphisms' function.")
    # raise NotImplementedError("sample_using_symmetric_prior requires `automorphisms` and further logic.")
    return jnp.eye(S_similarity_matrix.shape[0]) # Placeholder: returns identity matrix


# The __main__ block from the original file is moved to tests or examples.
# It contained calls to generate_latent_mdp and find_symmetric_mdp.
# The BUG comment about n_states=4/5 in find_symmetric_mdp should be investigated in tests.
