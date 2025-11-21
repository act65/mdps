import jax.numpy as np
from jax import grad, jacrev, jit
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
    Computes the mean and variance of the return distribution Z(s).
    Z(s) = R(s, A) + gamma * Z(S')
    
    Args:
        P (np.ndarray): the transition tensor [n_states, n_states, n_actions]
        r (np.ndarray): the reward matrix [n_states, n_actions]
        pi (np.ndarray): the policy [n_states, n_actions]
        discount (float): the discount rate

    Returns:
        mean (np.ndarray): Expected value V(s) [n_states]
        variance (np.ndarray): Variance of return [n_states]
    """
    n_states = P.shape[0]
    
    # 1. Compute Mean (Standard Value Function)
    # P_pi[s, s'] = sum_a pi(s, a) P(s, s', a)
    P_pi = np.einsum('ijk,jk->ij', P, pi)
    # r_pi[s] = sum_a pi(s, a) r(s, a)
    r_pi = np.einsum('jk,jk->j', pi, r)
    
    # V = (I - gamma P_pi)^-1 r_pi
    # P_pi is [next, curr], so we need P_pi.T for value propagation
    # V(s) = r(s) + gamma * sum_s' P(s'|s) V(s')
    I = np.eye(n_states)
    V = np.linalg.solve(I - discount * P_pi.T, r_pi)
    
    # 2. Compute Second Moment
    # M2 = E[Z^2]
    # M2(s) = E[ (r + gamma Z(s'))^2 ]
    #       = E[ r^2 + 2 gamma r Z(s') + gamma^2 Z(s')^2 ]
    #       = sum_a pi(s,a) [ r(s,a)^2 + 2 gamma r(s,a) E[Z(s')|s,a] + gamma^2 E[Z(s')^2|s,a] ]
    #       = sum_a pi(s,a) r(s,a)^2 
    #         + 2 gamma sum_a pi(s,a) r(s,a) sum_s' P(s'|s,a) V(s')
    #         + gamma^2 sum_a pi(s,a) sum_s' P(s'|s,a) M2(s')
    
    # Term 1: E[r^2]
    r2_pi = np.einsum('jk,jk->j', pi, r**2)
    
    # Term 2: 2 gamma E[r * E[Z(s')]]
    # E[Z(s')|s,a] = sum_s' P(s'|s,a) V(s')
    EZ_next = np.einsum('ijk,i->jk', P, V) # [n_states, n_actions]
    cross_term = 2 * discount * np.einsum('jk,jk,jk->j', pi, r, EZ_next)
    
    # RHS constant part
    b_m2 = r2_pi + cross_term
    
    # Solve for M2: (I - gamma^2 P_pi) M2 = b_m2
    # Again, use P_pi.T
    M2 = np.linalg.solve(I - discount**2 * P_pi.T, b_m2)
    
    # Variance = M2 - V^2
    Variance = M2 - V**2
    
    return V, Variance
