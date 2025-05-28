import jax.numpy as jnp
from jax import jit
from numpy.typing import NDArray
from typing import Callable

import mdp.utils as utils # Assuming utils.MDP and utils.bellman_optimality_operator are defined here

# Neural tangent kernel and related concepts for generalization in RL.
# Inspired by "Towards Characterizing Divergence in Deep Q-Learning" (https://arxiv.org/abs/1903.08894)

def adjusted_value_iteration(
    mdp_object: utils.MDP, 
    learning_rate: float, 
    D_matrix: NDArray[jnp.float_], 
    K_matrix: NDArray[jnp.float_]
) -> Callable[[NDArray[jnp.float_]], NDArray[jnp.float_]]:
    """
    Returns an update function for an "adjusted" value iteration process.
    This update rule incorporates matrices D and K, which might represent
    some form of kernel, basis transformation, or adjustment based on state similarity
    or feature representation, in the spirit of Neural Tangent Kernels or similar concepts.

    The update rule is: Q_next = Q_current + learning_rate * K @ D @ (T_bellman(Q_current) - Q_current)
    where T_bellman is the Bellman optimality operator.

    Args:
        mdp_object: The MDP environment, instance of `utils.MDP`.
        learning_rate: The learning rate (alpha) for the updates.
        D_matrix: A matrix used in the adjustment (e.g., state feature matrix, diagonal matrix for weighting).
                  Shape should be compatible for `D @ TD_error`. If TD_error is (S,A), D could be (S,S) or (A,A)
                  or more complex depending on its role. Assuming (S,S) if K is (S,S) and error is V-error (S,1),
                  or if error is Q-error (S,A), D and K would need to handle (S,A) shaped errors.
                  Given T(Q)-Q is (S,A), D could be (A,A) and K (A,A) for example, or (S,S) if applied to V-error.
                  The original paper might focus on V-values, so (S,1) error.
                  Let's assume Q is (S,A) and T(Q)-Q is (S,A). For np.dot(D, T(Q)-Q) to work,
                  if D is (X, S*A), then error needs to be flattened. Or D acts on rows/cols.
                  Most likely, if Q is (S,A), K and D are matrices that can operate on this.
                  E.g., if K and D are (S,S) and applied to V-values (S,1).
                  If operating on Q values (S,A), K and D might be more complex (e.g. Kronecker products)
                  or this is a simplified notation where dot implies broadcasting or element-wise multiplication.
                  For now, assuming Q is (S,A) and K, D are compatible for matrix multiplication
                  if Q is flattened, or they operate per-state/action.
                  If T(Q)-Q is (S,A), and K, D are (S,S), this implies a state-based adjustment.
                  Let's assume K and D are (S,S) and the operation is on V-values derived from Q.
                  Or, if Q is (S,A), K and D are (A,A) and operate on Q per state (less likely).
                  The most direct interpretation if Q is (S,A) and K,D are (S,S):
                  K @ D @ (T(Q)-Q) implies (T(Q)-Q) is treated as a collection of S vectors of dim A,
                  and K@D operates on the state dimension. Or (T(Q)-Q) is flattened.
                  Let's assume T(Q)-Q is (S,A) and K, D are (S,S), meaning dot operates on the first dim.
                  K @ D @ Error_SA -> K_SS @ D_SS @ Error_SA -> (S,A) output.
                  This is standard if Error_SA is treated as S vectors of size A, and K,D operate on states.

        K_matrix: Another matrix used in the adjustment (e.g., kernel matrix). Shape similar to D_matrix.

    Returns:
        A JIT-compiled callable update function `U(Q_current) -> Q_next`.
    """
    
    # T_bellman is the Bellman optimality operator for Q-values.
    T_bellman_q = lambda Q_values: utils.bellman_optimality_operator(
        mdp_object.P, mdp_object.r, Q_values, mdp_object.discount
    )

    @jit
    def update_rule_adjusted_q(Q_current: NDArray[jnp.float_]) -> NDArray[jnp.float_]:
        td_error: NDArray[jnp.float_] = T_bellman_q(Q_current) - Q_current # Shape (S, A)
        
        # Adjustment term: K @ D @ td_error
        # If K, D are (S,S) and td_error is (S,A):
        # D_times_td_error = D_matrix @ td_error  (S,S) @ (S,A) -> (S,A)
        # K_times_adjustment = K_matrix @ D_times_td_error (S,S) @ (S,A) -> (S,A)
        adjustment: NDArray[jnp.float_] = jnp.dot(K_matrix, jnp.dot(D_matrix, td_error))
        
        Q_next: NDArray[jnp.float_] = Q_current + learning_rate * adjustment
        return Q_next

    return update_rule_adjusted_q

# Commented out function from original file:
# def corrected_value_iteration(mdp, lr):
#     # This seems to be an outline for a natural policy gradient or similar method,
#     # involving Fisher Information Matrix (K_matrix = (dQdw.T @ dQdw)^-1 )
#     # Q(w) implies Q is parameterized by w.
#     # T = lambda theta: mdp.r + mdp.discount * np.argmax(mdp.P * Q(w)) # This T is not standard Bellman.
#     dQdw = lambda w: grad(Q) # Q needs to be defined as a function of w
#     Km1 = lambda w: np.linalg.inv(np.dot(dQdw(w).T, dQdw(w)))
#     U = lambda w: w + lr * np.dot(dQdw(w), np.dot(Km1, T(Q(w) - Q(w)))) # T(Q(w)-Q(w)) is problematic.
#     return U
