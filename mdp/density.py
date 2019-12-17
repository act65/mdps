import jax.numpy as np
import mdp.utils as utils

def density_value_functional(p_pi, P, r, pi, discount):
    """
    Args:
        px (float): the probability of pi
        P (np.ndarray): the transition tensor [n_states, n_states, n_actions]
        r (np.ndarray): the reward matrix [n_states, n_actions]
        pi (np.ndarray): the policy [n_states, n_actions]
        discount (float): the discount rate

    Returns:
        (np.ndarray): the ???.
    """
    P_pi = np.einsum('ijk,jk->ij', P, pi) #np.dot(M_pi, P)
    r_pi = np.einsum('jk,jk->j', pi, r)  #np.dot(M_pi, r)

    J = value_jacobian(r_pi, P_pi, discount)
    return probability_chain_rule(p_pi, J)

def value_jacobian(r_pi, P_pi, discount):
    """
    Jacobian of the value functional wrt the policy.

    Args:
        r_pi (np.ndarray): [n_states, 1]
        P_pi (np.ndarray): [n_states, n_states]
        discount (scalar): the discount rate

    Returns:
        [inputs x outputs]
    """
    return r_pi * (np.eye(P_pi.shape[0]) - discount * P_pi)**(-2)

def probability_chain_rule(px, J):
    """
    p(f(x)) = abs(|J|)^-1 . p(x)
    """
    return (np.abs(np.linalg.det(J))**(-1)) * px

def entropy_jacobian(pi):
    """
    H(pi) = - sum p log p
    dHdpi(j) = 1 + log p
    """
    return -1 - np.log(pi)

def distributional_value_function(P, r, pi, discount):
    """
    Value_functional: \Delta(\Pi) \times \Delta(P) \to \Delta(V)
    ??!?

    V(s) = E_{a\sim \pi(\cdot|s)} \bigg[ r(s, a) + \gamma E_{s' \sim P(\cdot|s, a)}[V(s')]  \bigg]

    r_pi(s) = \sum_a r(s, a) . pi(a|s)
    p(r | s, a) = r(s, a) . pi(a|s)

    P_pi(s) = \sum_a P(s'| s, a) . pi(a|s)
    p(s' | s, a) = r(s, a) . pi(a|s)


    Args:
        P (np.ndarray): the transition tensor [n_states, n_states, n_actions]
        r (np.ndarray): the reward matrix [n_states, n_actions]
        pi (np.ndarray): the policy [n_states, n_actions]
        discount (float): the discount rate

    Returns:
        (np.ndarray): the ???.
    """
    pass
