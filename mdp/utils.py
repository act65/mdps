import collections # For original MDP namedtuple, will be replaced by typing.NamedTuple
import itertools
import typing
from typing import List, Tuple, Any, Callable, Union, Sequence, Optional # Added Optional

import jax.numpy as jnp # Renamed for clarity and consistency
from jax import jit
import numpy.random as rnd # Standard numpy random for now
import numpy # For type hints and potentially some operations if needed separately from JAX

import mdp.search_spaces as search_spaces # Assuming this is a valid local module

# --- Type Definitions ---
NDArrayJax = jnp.ndarray # Alias for JAX ndarray for type hints
NDArrayNumpy = numpy.ndarray # Alias for NumPy ndarray

class MDP(typing.NamedTuple):
    """
    Represents a Markov Decision Process (MDP).

    Attributes:
        S (int): Number of states.
        A (int): Number of actions.
        P (NDArrayJax): Transition probability tensor of shape (S_next, S_current, A).
                      P[s_next, s_current, a] is the probability of transitioning to state `s_next`
                      from state `s_current` by taking action `a`.
                      Note: This (destination, source, action) convention is common in some contexts
                      but differs from (source, action, destination) used in others (e.g. Sutton & Barto).
                      The code using this namedtuple must be consistent with this convention.
        r (NDArrayJax): Reward matrix of shape (S, A). r[s, a] is the expected reward
                      for taking action `a` in state `s`.
        discount (float): Discount factor (gamma), between 0 and 1.
        d0 (NDArrayJax): Initial state distribution of shape (S, 1) or (S,).
                       d0[s] is the probability of starting in state `s`.
    """
    S: int
    A: int
    P: NDArrayJax
    r: NDArrayJax
    discount: float
    d0: NDArrayJax


# --- Utility Functions ---

def onehot(index: int, num_classes: int) -> NDArrayJax:
    """
    Creates a one-hot encoded vector.

    Args:
        index: The index to be set to 1.
        num_classes: The total number of classes (length of the one-hot vector).

    Returns:
        A JAX array representing the one-hot vector of shape (num_classes,).
    """
    return jnp.eye(num_classes, dtype=jnp.float32)[index]

def entropy(probabilities: NDArrayJax, epsilon: float = 1e-8) -> NDArrayJax:
    """
    Calculates the Shannon entropy of a probability distribution.
    H(p) = - sum(p_i * log(p_i))

    Args:
        probabilities: A JAX array representing a probability distribution.
                       Elements should be non-negative and sum to 1.
        epsilon: A small constant to prevent log(0).

    Returns:
        A scalar JAX array representing the entropy.
    """
    return -jnp.sum(jnp.log(probabilities + epsilon) * probabilities)

def sigmoid(x: NDArrayJax) -> NDArrayJax:
    """
    Computes the sigmoid function element-wise.
    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input JAX array.

    Returns:
        A JAX array with the sigmoid function applied element-wise.
    """
    return 1 / (1 + jnp.exp(-x))

def softmax(x: NDArrayJax, axis: int = -1, epsilon: float = 1e-8) -> NDArrayJax:
    """
    Computes the softmax function along a specified axis.
    softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

    Args:
        x: Input JAX array.
        axis: The axis along which the softmax is computed.
        epsilon: A small constant added to the denominator for numerical stability,
                 though usually max-subtraction handles this better. This epsilon might
                 not be standard for softmax.

    Returns:
        A JAX array with the softmax function applied.
    """
    exp_x = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True)) # Max subtraction for stability
    return exp_x / (jnp.sum(exp_x, axis=axis, keepdims=True) + epsilon)


def normalize(matrix: NDArrayJax, axis: int = 1, epsilon: float = 1e-8) -> NDArrayJax:
    """
    Normalizes vectors along a specified axis using L2 norm.

    Args:
        matrix: Input JAX array.
        axis: The axis along which to normalize. For rows, axis=1. For columns, axis=0.
        epsilon: A small constant to prevent division by zero if a vector's norm is zero.

    Returns:
        A JAX array with vectors normalized along the specified axis.
    """
    magnitudes: NDArrayJax = jnp.linalg.norm(matrix, axis=axis, keepdims=True)
    return matrix / (magnitudes + epsilon)

def clip_by_norm(vector: NDArrayJax, max_norm: float, epsilon: float = 1e-8) -> NDArrayJax:
    """
    Clips a vector such that its L2 norm does not exceed `max_norm`.
    If norm(vector) > max_norm, scales it down to have norm `max_norm`.
    The original implementation had a sigmoid-based weighted average, which is unusual.
    This version implements standard norm clipping.

    Args:
        vector: Input JAX array (vector).
        max_norm: The maximum allowable L2 norm.
        epsilon: Small constant for numerical stability if vector norm is zero.

    Returns:
        The clipped JAX array (vector).
    """
    current_norm: NDArrayJax = jnp.linalg.norm(vector)
    # Standard clipping:
    if current_norm > max_norm:
        return vector * (max_norm / (current_norm + epsilon))
    return vector
    # Original logic:
    # p_val = sigmoid(current_norm - max_norm)
    # return p_val * vector * max_norm / (current_norm + epsilon) + (1 - p_val) * vector


# --- MDP Construction ---

def build_random_mdp(num_states: int, num_actions: int, discount_factor: float) -> MDP:
    """
    Builds a random MDP with specified dimensions and discount factor.
    Transition probabilities P(s'|s,a) (P[s',s,a] in code) are normalized over s'.
    Initial state distribution d0 is normalized.

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A).
        discount_factor: Discount factor (gamma).

    Returns:
        An MDP namedtuple instance.
    """
    # Using numpy.random for generation, then converting to JAX array.
    P_np: NDArrayNumpy = rnd.random((num_states, num_states, num_actions)) # P[s_next, s_current, action]
    P_normalized_np: NDArrayNumpy = P_np / P_np.sum(axis=0, keepdims=True) # Normalize over s_next for each (s_current, action)
    
    r_np: NDArrayNumpy = rnd.standard_normal((num_states, num_actions)) # r[s_current, action]
    
    d0_np: NDArrayNumpy = rnd.random((num_states, 1)) # d0[s_current]
    d0_normalized_np: NDArrayNumpy = d0_np / d0_np.sum(axis=0, keepdims=True)
    
    return MDP(
        S=num_states, 
        A=num_actions, 
        P=jnp.array(P_normalized_np), 
        r=jnp.array(r_np), 
        discount=discount_factor, 
        d0=jnp.array(d0_normalized_np)
    )

def sparsify(array: NDArrayNumpy, sparsity_factor: float = 0.5) -> NDArrayNumpy:
    """
    Sparsifies a NumPy array by randomly setting elements to zero.

    Args:
        array: The input NumPy array.
        sparsity_factor: The approximate fraction of elements to be zeroed out.
                         (1 - sparsity_factor) is the density.

    Returns:
        A sparsified NumPy array with the same dtype as the input.
    """
    mask: NDArrayNumpy = rnd.random(array.shape) > sparsity_factor
    return array * mask.astype(array.dtype)

def build_random_sparse_mdp(num_states: int, num_actions: int, discount_factor: float, sparsity: float = 0.5) -> MDP:
    """
    Builds a random MDP with sparse transition and reward matrices.

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A).
        discount_factor: Discount factor (gamma).
        sparsity: The sparsity factor for P and r matrices (fraction of zeros).

    Returns:
        An MDP namedtuple instance with sparse P and r.
    """
    P_np: NDArrayNumpy = sparsify(rnd.random((num_states, num_states, num_actions)), sparsity)
    r_np: NDArrayNumpy = sparsify(rnd.standard_normal((num_states, num_actions)), sparsity)
    d0_np: NDArrayNumpy = rnd.random((num_states, 1))

    # Normalize P: sum over s_next should be 1 for each (s_current, action)
    # Handle cases where a column might be all zeros after sparsification
    P_sum_np: NDArrayNumpy = P_np.sum(axis=0, keepdims=True)
    # Avoid division by zero: if sum is 0, make it uniform (or handle as error/re-sparsify)
    P_sum_np[P_sum_np == 0] = 1.0 # Replace 0 sums with 1 to avoid NaN, effectively making P(s'|s,a)=0 for that (s,a)
                                # A better strategy might be to add small epsilon or re-generate.
    P_normalized_np: NDArrayNumpy = P_np / P_sum_np
    
    d0_normalized_np: NDArrayNumpy = d0_np / d0_np.sum(axis=0, keepdims=True)
    
    return MDP(
        S=num_states, 
        A=num_actions, 
        P=jnp.array(P_normalized_np), 
        r=jnp.array(r_np), 
        discount=discount_factor, 
        d0=jnp.array(d0_normalized_np)
    )


# --- Policy Generation ---

def gen_grid_policies(num_points_per_dim: int, num_states: int = 2, num_actions: int = 2) -> List[NDArrayJax]:
    """
    Generates a grid of policies for a 2-state, 2-action MDP, or more generally by iterating probabilities.
    For S=2, A=2: policy is [[p1, 1-p1], [p2, 1-p2]]. This function creates a grid for p1, p2.
    The current implementation is specific to S=2, A=2.
    TODO: Generalize for arbitrary S, A if the "grid" concept applies.

    Args:
        num_points_per_dim: Number of points to discretize each probability dimension (e.g., p1, p2).
        num_states: Number of states (currently hardcoded logic for S=2).
        num_actions: Number of actions (currently hardcoded logic for A=2).

    Returns:
        A list of policy JAX arrays. Each policy is shape (num_states, num_actions).
    """
    if num_states != 2 or num_actions != 2:
        # Fallback or error for non-2x2 cases, as original TODO noted.
        # For now, let's stick to the 2x2 implementation.
        # A general version would involve iterating over simplex dimensions for each state.
        raise NotImplementedError("gen_grid_policies is currently implemented only for S=2, A=2.")

    # Probabilities for action 0 in state 0 (p1) and state 1 (p2)
    probabilities_dim1: NDArrayJax = jnp.linspace(0, 1, num_points_per_dim)
    probabilities_dim2: NDArrayJax = jnp.linspace(0, 1, num_points_per_dim)
    
    policies: List[NDArrayJax] = []
    for p1_val in probabilities_dim1: # Renamed p1 to p1_val
        for p2_val in probabilities_dim2: # Renamed p2 to p2_val
            policy = jnp.array([
                [p1_val, 1 - p1_val],
                [p2_val, 1 - p2_val] # Original had [1-p2_val, p2_val], assuming p2_val is for action 0 in state 1.
            ], dtype=jnp.float32)
            policies.append(policy)
    return policies

def get_deterministic_policies(num_states: int, num_actions: int) -> List[NDArrayJax]:
    """
    Generates all possible deterministic policies for an MDP.
    A deterministic policy assigns one action to each state.

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A).

    Returns:
        A list of all deterministic policies. Each policy is a JAX array of shape (S, A),
        one-hot encoded for actions. Total number of policies is A^S.
    """
    # Create one-hot vectors for each action
    action_onehots: List[NDArrayJax] = [onehot(i, num_actions) for i in range(num_actions)]
    
    # Generate all combinations of actions for each state
    # itertools.product creates tuples of one-hot action vectors, one for each state
    policy_action_choices_tuples: List[Tuple[NDArrayJax, ...]] = list(
        itertools.product(*([action_onehots] * num_states))
    )
    
    # Stack the tuples of action vectors to form policy matrices
    policies: List[NDArrayJax] = [jnp.stack(p_tuple) for p_tuple in policy_action_choices_tuples]
    return policies

def get_random_policy_2x2() -> NDArrayJax:
    """
    Generates a random stochastic policy for a 2-state, 2-action MDP.
    Policy is [[p1, 1-p1], [p2, 1-p2]] where p1, p2 are random.

    Returns:
        A JAX array of shape (2, 2) representing the random policy.
    """
    p1_prob_action0_state0: float = rnd.random() # Probability of action 0 in state 0
    p2_prob_action0_state1: float = rnd.random() # Probability of action 0 in state 1
    return jnp.array([
        [p1_prob_action0_state0, 1 - p1_prob_action0_state0],
        [p2_prob_action0_state1, 1 - p2_prob_action0_state1]
    ], dtype=jnp.float32)

def rnd_simplex(dimension: int) -> NDArrayNumpy: # Returns NumPy array as per original
    """
    Generates a random point from a simplex of a given dimension.
    A point on a simplex is a vector of non-negative numbers that sum to 1.

    Args:
        dimension: The dimension of the simplex (number of elements in the vector).

    Returns:
        A NumPy array of shape (dimension,) representing a point on the simplex.
    """
    if dimension <= 0:
        return numpy.array([], dtype=numpy.float64)
    if dimension == 1:
        return numpy.array([1.0], dtype=numpy.float64)
        
    # Generate `dimension - 1` random points in [0,1], sort them,
    # then take differences between sorted points (and 0, 1 boundaries).
    # This is a standard way to sample from a simplex.
    pts: NDArrayNumpy = rnd.uniform(0, 1, dimension - 1)
    sorted_pts: List[float] = sorted(list(pts))
    diffs: NDArrayNumpy = numpy.diff([0.0] + sorted_pts + [1.0])
    return diffs.astype(numpy.float64)

def random_policy(num_states: int, num_actions: int) -> NDArrayJax:
    """
    Generates a random stochastic policy.
    For each state, action probabilities are drawn from a simplex.

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A).

    Returns:
        A JAX array of shape (S, A) representing the random policy.
    """
    if num_actions == 0: # Handle edge case
        return jnp.empty((num_states, 0), dtype=jnp.float32)
    policy_rows: List[NDArrayNumpy] = [rnd_simplex(num_actions) for _ in range(num_states)]
    return jnp.array(numpy.vstack(policy_rows), dtype=jnp.float32)

def random_det_policy(num_states: int, num_actions: int) -> NDArrayJax:
    """
    Generates a random deterministic policy.
    For each state, one action is chosen uniformly at random.

    Args:
        num_states: Number of states (S).
        num_actions: Number of actions (A).

    Returns:
        A JAX array of shape (S, A), one-hot encoded, representing the random deterministic policy.
    """
    if num_actions == 0: # Handle edge case
        return jnp.empty((num_states, 0), dtype=jnp.float32)
    policy_rows: List[NDArrayJax] = [
        onehot(rnd.randint(0, num_actions), num_actions) for _ in range(num_states)
    ]
    return jnp.vstack(policy_rows)


# --- Value Function and Bellman Operators ---

# @jit # Jitting might be problematic if list comprehensions or certain Python features are used inside.
def polytope(
    P_tensor: NDArrayJax, 
    r_matrix: NDArrayJax, 
    discount_factor: float, 
    policy_list: List[NDArrayJax]
) -> NDArrayJax:
    """
    Computes the value function for each policy in a list and stacks them.
    The sum over axis 1 in the original code `np.sum(value_functional(...), axis=1)`
    is unusual if `value_functional` returns V(s) of shape (S,1) or (S,).
    Summing V(s) over states is not standard for constructing a polytope of value functions.
    Usually, each V(s) vector itself is a point, or the polytope is in V-space.
    Assuming the intent was to get the sum of state values for each policy (a scalar per policy).
    If V(s) is (S,1), sum(axis=1) on (S,1) is not meaningful. If (S,), sum is scalar.
    This function seems to aim to create a matrix where each row is related to a policy's value.
    If the goal is a matrix of value functions (N_pis, S):

    Args:
        P_tensor: Transition probability tensor (S_next, S_current, A).
        r_matrix: Reward matrix (S, A).
        discount_factor: Discount factor (gamma).
        policy_list: A list of policies. Each policy is (S, A).

    Returns:
        A JAX array where each row is the value function V(s) for a policy, shape (num_policies, S).
    """
    value_functions: List[NDArrayJax] = [
        jnp.squeeze(value_functional(P_tensor, r_matrix, policy, discount_factor)) # Squeeze to (S,)
        for policy in policy_list
    ]
    return jnp.vstack(value_functions) # Stacks to (num_policies, S)


@jit
def value_functional(
    P_tensor: NDArrayJax, 
    r_matrix: NDArrayJax, 
    policy: NDArrayJax, 
    discount_factor: float
) -> NDArrayJax:
    """
    Computes the value function V_pi(s) for a given policy pi using the matrix inversion method.
    The value function is defined by the Bellman equation: V_pi = r_pi + gamma * P_pi * V_pi.
    This can be solved as V_pi = (I - gamma * P_pi)^-1 * r_pi.

    P_pi[s, s'] = sum_a policy[s, a] * P_tensor[s', s, a]
        This is the probability of transitioning from state `s` to state `s'` under policy `pi`.
        `s` is the current state (row index for P_pi), `s'` is the next state (column index for P_pi).
    r_pi[s] = sum_a policy[s, a] * r_matrix[s, a]
        This is the expected reward in state `s` under policy `pi`.

    Args:
        P_tensor: Transition probability tensor, shape (S_next, S_current, A).
                  P_tensor[s_next, s_current, a] = P(s_next | s_current, a).
        r_matrix: Reward matrix, shape (S_current, A). r_matrix[s, a] = R(s, a).
        policy: Policy matrix, shape (S_current, A). policy[s, a] = pi(a|s).
        discount_factor: Discount factor (gamma).

    Returns:
        Value function V_pi as a JAX array of shape (S_current, 1).
    """
    num_states: int = P_tensor.shape[1] # S_current

    # P_policy[s_current, s_next] = sum_a policy[s_current, a] * P_tensor[s_next, s_current, a]
    P_policy: NDArrayJax = jnp.einsum('sa,nsa->sn', policy, P_tensor) # s=current,a=action,n=next
                                                                    # policy(s,a), P(n,s,a) -> P_policy(s,n)
    
    # r_policy[s_current] = sum_a policy[s_current, a] * r_matrix[s_current, a]
    r_policy: NDArrayJax = jnp.einsum('sa,sa->s', policy, r_matrix)[:, jnp.newaxis] # Result (S,1)

    # V = (I - gamma * P_policy)^-1 * r_policy.
    # The original code had P_pi.T. Let's trace the dimensions:
    # Original P_pi = jnp.einsum('ijk,jk->ij', P, pi)
    # If P was (S_next, S_curr, A) `ijk` and pi was (S_curr, A) `jk`, then P_pi would be (S_next, S_curr) `ik`.
    # P_pi[s_next, s_curr].
    # Then P_pi.T would be P_pi_T[s_curr, s_next]. This is the standard P_pi(s,s') used in (I - gP_pi)V = r_pi.
    # My `P_policy` is already P_policy[s_curr, s_next], so no transpose is needed here.
    
    identity_matrix: NDArrayJax = jnp.eye(num_states)
    matrix_to_invert: NDArrayJax = identity_matrix - discount_factor * P_policy
    
    state_values: NDArrayJax = jnp.linalg.solve(matrix_to_invert, r_policy) # Solves for V in (I-gP_pi)V = r_pi
    return state_values


def bellman_optimality_operator(
    P_tensor: NDArrayJax, 
    r_matrix: NDArrayJax, 
    Q_current: NDArrayJax, 
    discount_factor: float
) -> NDArrayJax:
    """
    Applies one step of the Bellman optimality operator for Q-values.
    Q_next(s, a) = r(s, a) + gamma * sum_{s'} P(s'|s,a) * max_{a'} Q_current(s', a')

    Args:
        P_tensor: Transition probability tensor, shape (S_next, S_current, A).
                  P_tensor[s_next, s_current, a] = P(s_next | s_current, a).
        r_matrix: Reward matrix, shape (S_current, A). r_matrix[s, a] = R(s, a).
        Q_current: Current Q-value function, shape (S_current, A). Q_current[s',a'] for next state s'.
        discount_factor: Discount factor (gamma).

    Returns:
        Next Q-value function Q_next(s,a) as a JAX array of shape (S_current, A).
    """
    if not (len(Q_current.shape) == 2 and Q_current.shape[1] > 0): # Allow A=1
        raise ValueError("Q_current must be (S,A) and A > 0")


    # V_max_current[s_next] = max_a' Q_current(s_next, a')
    # Q_current is indexed by (state, action). If it's Q(s',a'), then states are s_next.
    V_max_at_s_next: NDArrayJax = jnp.max(Q_current, axis=1) # Shape (S_next,) assuming Q_current is Q(s_next, a')
    
    # Expected future value: E_val[s_current,a] = sum_{s_next} P(s_next|s_current,a) * V_max_at_s_next[s_next]
    # P_tensor[s_next, s_current, a]
    # einsum: P(n,s,a) * V_max(n) -> EV(s,a)
    expected_future_value: NDArrayJax = jnp.einsum('nsa,n->sa', P_tensor, V_max_at_s_next) # (S_current, A)
    
    return r_matrix + discount_factor * expected_future_value


def bellman_operator(
    P_tensor: NDArrayJax, 
    r_matrix: NDArrayJax, 
    V_current_s_next: NDArrayJax, # Renamed to clarify it's V(s')
    discount_factor: float
) -> NDArrayJax:
    """
    Applies one step of the Bellman operator for state-values V to get action-values Q.
    Q_next(s, a) = r(s, a) + gamma * sum_{s'} P(s'|s,a) * V_current_s_next(s')

    Args:
        P_tensor: Transition probability tensor, shape (S_next, S_current, A).
                  P_tensor[s_next, s_current, a] = P(s_next | s_current, a).
        r_matrix: Reward matrix, shape (S_current, A). r_matrix[s, a] = R(s, a).
        V_current_s_next: Current value function V(s'), shape (S_next,) or (S_next, 1).
        discount_factor: Discount factor (gamma).

    Returns:
        Action-value function Q(s,a) as a JAX array of shape (S_current, A).
    """
    V_squeezed_s_next: NDArrayJax = jnp.squeeze(V_current_s_next) # Ensure V_current_s_next is (S_next,)
    
    # Expected future value: E_val[s,a] = sum_{s'} P(s'|s,a) * V_current_s_next[s']
    # P_tensor[s_next, s_current, a] * V_current_s_next[s_next] -> sum over s_next
    # einsum: P(n,s,a) * V(n) -> EV(s,a)
    expected_future_value: NDArrayJax = jnp.einsum('nsa,n->sa', P_tensor, V_squeezed_s_next) # (S_current, A)
    
    return r_matrix + discount_factor * expected_future_value

# --- Iterative Solving Utilities ---

def isclose(
    val_x: Any, 
    val_y: Any, 
    absolute_tolerance: float = 1e-8
) -> bool:
    """
    Checks if two values (or structures of values) are close, element-wise for arrays.
    Handles JAX arrays, lists/tuples of JAX arrays, and potentially nested structures
    if `search_spaces.build` can flatten them into comparable arrays.

    Args:
        val_x: First value or structure.
        val_y: Second value or structure.
        absolute_tolerance: The absolute tolerance for comparison.

    Returns:
        True if all corresponding elements are close, False otherwise.

    Raises:
        ValueError: If the input types are not supported or cannot be compared.
    """
    if isinstance(val_x, jnp.ndarray) and isinstance(val_y, jnp.ndarray):
        return jnp.allclose(val_x, val_y, atol=absolute_tolerance)
    elif isinstance(val_x, list) and isinstance(val_y, list):
        # This relies on search_spaces.build to convert lists (potentially of arrays)
        # into single JAX arrays for comparison. This might be too specific.
        # A more general list comparison would iterate and call isclose recursively.
        # For now, matching original logic.
        if not val_x or not val_y: # Handle empty lists
            return len(val_x) == len(val_y)
        try: # Assuming search_spaces.build handles lists of arrays
            return jnp.allclose(search_spaces.build(val_x), search_spaces.build(val_y), atol=absolute_tolerance)
        except Exception: # Fallback or error if build fails or types are mixed
            if len(val_x) != len(val_y): return False
            return all(isclose(ix, iy, absolute_tolerance) for ix, iy in zip(val_x, val_y))
    elif isinstance(val_x, tuple) and isinstance(val_y, tuple):
        if not val_x or not val_y: # Handle empty tuples
             return len(val_x) == len(val_y)
        # Check if elements are arrays or lists (as per original logic)
        if val_x and val_y and isinstance(val_x[0], jnp.ndarray) and isinstance(val_y[0], jnp.ndarray): # Added val_x and val_y check
            # Compare first elements if they are arrays (original logic was just x[0], y[0])
            # To compare all elements in tuple:
            if len(val_x) != len(val_y): return False
            return all(isclose(ix, iy, absolute_tolerance) for ix, iy in zip(val_x, val_y))
        elif val_x and val_y and isinstance(val_x[0], list) and isinstance(val_y[0], list): # Added val_x and val_y check
             # Again, relies on search_spaces.build for list elements within tuple
            if len(val_x) != len(val_y): return False
            return all(isclose(ix, iy, absolute_tolerance) for ix, iy in zip(val_x, val_y))
        else: # General tuple comparison
            if len(val_x) != len(val_y): return False
            return all(isclose(ix, iy, absolute_tolerance) for ix, iy in zip(val_x, val_y))
    elif isinstance(val_x, (int, float, numpy.number, jnp.number)) and isinstance(val_y, (int, float, numpy.number, jnp.number)): # Added numpy and jax number types
        return jnp.isclose(jnp.array(val_x), jnp.array(val_y), atol=absolute_tolerance).item()
    else:
        raise ValueError(f"Unsupported types for isclose: {type(val_x)}, {type(val_y)}")


def converged(
    value_history: List[Any], 
    min_iterations_for_check: int = 1, # Allow checking from the second iteration
    iteration_threshold_low_tol: int = 5000, low_tol: float = 1e-6,
    iteration_threshold_med_tol: int = 10000, med_tol: float = 1e-4,
    iteration_threshold_high_tol: int = 1, high_tol: float = 1e-8, # Strictest tol from 1st check
    max_iterations_error: int = 20000
) -> bool:
    """
    Checks for convergence in a list of historical values from an iterative process.
    Convergence is determined if the last two values are close, with varying tolerances
    based on the number of iterations.

    Args:
        value_history: A list of values, where each element is a state/result from an iteration.
        min_iterations_for_check: Minimum number of iterations before checking convergence.
        iteration_threshold_low_tol: Iteration count to start using `low_tol`.
        low_tol: Tolerance for convergence after `iteration_threshold_low_tol` iterations.
        iteration_threshold_med_tol: Iteration count to start using `med_tol`.
        med_tol: Tolerance for convergence after `iteration_threshold_med_tol` iterations.
        iteration_threshold_high_tol: Iteration count to start using `high_tol`.
        high_tol: Strictest tolerance, used after `iteration_threshold_high_tol` iterations.
        max_iterations_error: Maximum number of iterations before raising a ValueError if not converged.

    Returns:
        True if convergence criteria are met, False otherwise.

    Raises:
        ValueError: If `max_iterations_error` is reached without convergence or if NaNs are found.
    """
    if len(value_history) < min_iterations_for_check + 1: # Need at least two values to compare
        return False

    last_val = value_history[-1]
    prev_val = value_history[-2]

    # Check for NaNs (original code commented this out, but good to have)
    # This check needs to be robust for various types in last_val
    # For JAX arrays:
    if isinstance(last_val, jnp.ndarray) and jnp.isnan(last_val).any():
        raise ValueError('NaNs encountered in iteration history.')
    # Add more comprehensive NaN checks if history contains complex structures.

    # Convergence checks with varying tolerance based on iteration count
    current_iter_count = len(value_history)
    if current_iter_count > iteration_threshold_low_tol and isclose(last_val, prev_val, low_tol):
        return True
    if current_iter_count > iteration_threshold_med_tol and isclose(last_val, prev_val, med_tol):
        return True
    # Strictest tolerance applied earlier or as default
    if current_iter_count > iteration_threshold_high_tol and isclose(last_val, prev_val, high_tol):
        return True
    
    if current_iter_count > max_iterations_error:
        # print(value_history[-5:-1]) # Original debug print
        raise ValueError(f'Not converged after {max_iterations_error} iterations.')
        
    return False

def solve(
    update_function: Callable[[Any], Any], 
    initial_value: Any,
    max_iter: int = 20000, # Default max iterations from original converged
    tol: float = 1e-8,     # Default tolerance from original converged
    verbose: bool = False,
    print_every: int = 100
) -> List[Any]:
    """
    Generically solves for a fixed point of `update_function` starting from `initial_value`.
    Iteratively applies `x_next = update_function(x_current)` until convergence.

    Args:
        update_function: A function that takes the current value and returns the next value.
        initial_value: The starting value for the iteration.
        max_iter: Maximum number of iterations before raising an error.
        tol: Tolerance used for the `isclose` check in `converged`.
        verbose: If True, prints progress.
        print_every: How often to print progress if verbose is True.

    Returns:
        A list containing the history of values from each iteration, including the initial value.
    """
    iteration_history: List[Any] = [initial_value]
    current_value: Any = initial_value
    
    # Parameters for `converged` (can be exposed as args to `solve` if needed)
    # Using the max_iter and tol passed to solve for the primary convergence check.
    iter_thresh_low_tol = max(max_iter // 4, 100) 
    low_tol = tol * 100
    iter_thresh_med_tol = max(max_iter // 2, 200)
    med_tol = tol * 10

    while not converged(
        iteration_history, 
        iteration_threshold_high_tol=1, high_tol=tol, # Check strict tol from start
        iteration_threshold_low_tol=iter_thresh_low_tol, low_tol=low_tol,
        iteration_threshold_med_tol=iter_thresh_med_tol, med_tol=med_tol,
        max_iterations_error=max_iter
    ):
        current_value = update_function(current_value)
        iteration_history.append(current_value)
        if verbose and len(iteration_history) % print_every == 0:
            print(f'\rStep: {len(iteration_history)}', end='', flush=True)
    
    if verbose: print(f'\nConverged in {len(iteration_history)} steps.') # Added newline for better formatting
    return iteration_history

# --- Simulation and Sampling ---

@jit
def discounted_rewards(rewards_timeline: Union[List[float], NDArrayJax], discount_factor: float) -> NDArrayJax:
    """
    Calculates the discounted sum of rewards for a sequence of rewards.
    Uses (1-gamma) normalization factor if interpreting as average discounted reward,
    or remove it if simply sum_t gamma^t r_t is needed. Original included (1-gamma).

    Args:
        rewards_timeline: A list or JAX array of rewards received over time.
        discount_factor: Discount factor (gamma).

    Returns:
        A scalar JAX array representing the total discounted reward.
    """
    rewards_array: NDArrayJax = jnp.array(rewards_timeline)
    time_steps: NDArrayJax = jnp.arange(len(rewards_array))
    discounts: NDArrayJax = discount_factor ** time_steps
    # The (1-discount_factor) term normalizes if this is an infinite horizon average,
    # or if it's related to converting to an "effective" non-discounted sum.
    # For a finite sum of discounted rewards, it's typically not included.
    # Retaining it as per original code.
    return (1 - discount_factor) * jnp.sum(discounts * rewards_array)


@jit
def sample_from_distribution(
    probabilities: NDArrayJax, 
    temperature: float = 1.0,
    key: Optional[NDArrayJax] = None # For JAX explicit PRNG
) -> NDArrayJax:
    """
    Samples an index from a probability distribution using Gumbel-Max trick.
    This allows sampling from a categorical distribution.

    Args:
        probabilities: A 1D JAX array representing a probability distribution (sums to 1).
        temperature: Temperature parameter for Gumbel-Max. Higher temperature
                     leads to more uniform sampling, lower to more greedy.
                     temperature=0 would be argmax, but log(0) issues.
                     Small positive temperature for near-argmax.
        key: Optional JAX PRNG key for reproducible randomness. If None, uses numpy.random.
             For pure JAX, a key should be provided and split.

    Returns:
        A scalar JAX array representing the sampled index.
    """
    if key is not None:
        # TODO: Implement with jax.random.gumbel if this function needs to be pure JAX
        # gumbel_noise = jax.random.gumbel(key, shape=probabilities.shape) * temperature
        # For now, using numpy.random as in original for consistency
        pass

    # Gumbel-Max trick: argmax(log(p_i) + G_i) where G_i are Gumbel distributed.
    # G_i can be generated as -log(-log(U_i)) where U_i ~ Uniform(0,1).
    # Adding small epsilon to probabilities for log stability.
    epsilon: float = 1e-9
    if temperature == 0: # Deterministic case: argmax
        return jnp.argmax(probabilities, axis=-1)

    # Using numpy.random as per original code. For JAX-native, replace with jax.random.
    gumbel_noise_np: NDArrayNumpy = -numpy.log(-numpy.log(rnd.random(probabilities.shape) + epsilon) + epsilon)
    gumbel_noise: NDArrayJax = jnp.array(gumbel_noise_np) # Convert to JAX array

    perturbed_log_probs: NDArrayJax = jnp.log(probabilities + epsilon) + gumbel_noise * temperature
    sampled_index: NDArrayJax = jnp.argmax(perturbed_log_probs, axis=-1)
    return sampled_index


def rollout(
    # Assuming transition_fn P[s_next, s_current, action]
    transition_probabilities: NDArrayJax, # Shape (S_next, S_current, A)
    reward_function: NDArrayJax,          # Shape (S_current, A)
    initial_state_distribution: NDArrayJax, # Shape (S_current,) or (S_current, 1)
    policy: NDArrayJax,                   # Shape (S_current, A)
    num_timesteps: int,
    key: Optional[NDArrayJax] = None      # For JAX explicit PRNG if sample becomes pure JAX
) -> List[Tuple[int, int, float]]:
    """
    Performs a rollout (simulation) in an MDP for a given number of timesteps.

    Args:
        transition_probabilities: MDP transition tensor P(s'|s,a), shape (S_next, S_current, A).
        reward_function: MDP reward function R(s,a), shape (S_current, A).
        initial_state_distribution: Initial state distribution d0(s), shape (S_current,).
        policy: Policy pi(a|s) to follow, shape (S_current, A).
        num_timesteps: Number of timesteps (T) to simulate.
        key: Optional JAX PRNG key for reproducible sampling.

    Returns:
        A list of tuples (state, action, reward) representing the trajectory.
    """
    trajectory: List[Tuple[int, int, float]] = []
    
    # Sample initial state
    # current_state_idx: int = sample_from_distribution(jnp.squeeze(initial_state_distribution), key=key).item() # if key is used
    current_state_idx: int = sample_from_distribution(jnp.squeeze(initial_state_distribution)).item()


    for _ in range(num_timesteps):
        # Sample action based on policy in current state
        action_idx: int = sample_from_distribution(policy[current_state_idx, :]).item()
        
        # Get reward
        current_reward: float = reward_function[current_state_idx, action_idx].item()
        
        trajectory.append((current_state_idx, action_idx, current_reward))
        
        # Sample next state based on transition probabilities
        next_state_distribution: NDArrayJax = transition_probabilities[:, current_state_idx, action_idx]
        current_state_idx = sample_from_distribution(next_state_distribution).item()
        
    return trajectory
