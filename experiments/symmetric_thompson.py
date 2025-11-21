"""
Thompson sampling with a symmetric prior.
"""
import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap
import jax.random as jrandom

import mdp.utils as utils
import mdp.search_spaces as ss

# Global tolerance for similarity
TOL = 0.01

def complexity(mdp):
    """
    Complexity is measured by how many states are the same (under some measure of similarity).
    We use Q-value similarity.
    """
    # 1. Solve for Q*
    # We need a policy to evaluate or just use optimal Q?
    # Abstractions usually preserve optimal value or all values.
    # Let's use Q*
    
    # Initialize V
    V = np.zeros((mdp.S, 1))
    # Value Iteration
    try:
        V_star = utils.solve(lambda v: np.max(utils.bellman_operator(mdp.P, mdp.r, v, mdp.discount), axis=1, keepdims=True), V)[-1]
    except ValueError:
        # If not converged, return max complexity (worst case)
        return mdp.S
        
    Q_star = utils.bellman_operator(mdp.P, mdp.r, V_star, mdp.discount)
    
    # 2. Compute Similarity
    # |S| x |S| matrix
    # sim[i, j] = || Q*(i, :) - Q*(j, :) ||
    diffs = np.sum(np.abs(Q_star[:, None, :] - Q_star[None, :, :]), axis=-1)
    
    # 3. Count partitions
    # States are similar if diff < TOL
    similar = diffs < TOL
    
    m = np.max(np.sum(similar, axis=1))
    return mdp.S / m

def mdp_sampler(params):
    """
    Params -> MDP
    params: (P_logits, r)
    """
    P_logits, r = params
    n_states, _, n_actions = P_logits.shape
    
    # Use stable softmax from utils or jax.nn
    # utils.softmax is defined as exp(x)/sum(exp(x))
    # Let's use jax.nn.softmax for stability if available, or implement stable one
    # utils.softmax implementation: np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)
    # It defaults to axis -1. We need axis 0.
    
    # Stable softmax on axis 0
    max_logits = np.max(P_logits, axis=0, keepdims=True)
    exp_logits = np.exp(P_logits - max_logits)
    P = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    
    # r is [n_states, n_actions]
    
    # discount? fixed?
    discount = 0.9
    d0 = np.ones((n_states, 1)) / n_states
    
    return utils.MDP(n_states, n_actions, P, r, discount, d0)


def symmetric_sampler(params, rng_key):
    """
    Rejection sampling to find a 'simple' MDP close to params?
    Or just sample from params (which define a distribution) and reject based on complexity?
    
    If params is just a single MDP (mean), we need to sample around it?
    Or params define a distribution (e.g. logits + noise).
    """
    # For simplicity, let's assume params defines the mean, and we add noise to logits/rewards
    P_logits_mean, r_mean = params
    
    # We need a loop
    # But JAX loops are tricky.
    # Let's do a python loop since this is likely not JIT-ed top level
    
    for i in range(100): # Max attempts
        key, rng_key = jrandom.split(rng_key)
        
        # Add noise
        P_noise = jrandom.normal(key, P_logits_mean.shape) * 0.1
        r_noise = jrandom.normal(key, r_mean.shape) * 0.1
        
        sampled_params = (P_logits_mean + P_noise, r_mean + r_noise)
        m = mdp_sampler(sampled_params)
        
        # Check complexity
        # We want "simple" MDPs? Or "Symmetric" ones?
        # If complexity is low -> simple.
        # Let's say we accept if complexity < Threshold
        # Or prob proportional to 1/complexity?
        
        c = complexity(m)
        # Heuristic: accept if c < S/2
        if c < m.S / 1.5:
            return m
            
    return m # Return last one if failed

def mse(x, y):
    return np.mean((x - y)**2)

def thompson(true_mdp, lr=0.1):
    """
    Learn a model that matches the true MDP's value function,
    while favoring symmetric models via the sampler.
    """
    n_states = true_mdp.S
    n_actions = true_mdp.A
    
    # Initialize params
    P_logits = onp.random.normal(size=(n_states, n_states, n_actions))
    r = onp.random.normal(size=(n_states, n_actions))
    params = (P_logits, r)
    
    # Test policies for Value Equivalence
    pis = [utils.random_policy(n_states, n_actions) for _ in range(10)]
    pis = np.stack(pis) # [n_pis, n_states, n_actions]
    
    # True Values
    # vmap over policies
    get_val = vmap(lambda pi: utils.value_functional(true_mdp.P, true_mdp.r, pi, true_mdp.discount))
    V_true = get_val(pis) # [n_pis, n_states]
    
    # Gradient of Value Equivalence Loss
    def loss_fn(p):
        m = mdp_sampler(p)
        # We want V_model to match V_true
        V_model = vmap(lambda pi: utils.value_functional(m.P, m.r, pi, m.discount))(pis)
        return mse(V_true, V_model)
        
    dLdp = jit(grad(loss_fn))
    
    rng_key = jrandom.PRNGKey(0)

    def update_step(params, Q, rng_key):
        # 1. Sample a model from the "posterior" (approximated by rejection sampling around current params)
        # Actually, Thompson sampling usually means:
        # Sample M ~ P(M | Data).
        # Here we don't have Data, we have Value Equivalence objective.
        # And we maintain a point estimate 'params' and sample around it?
        
        # Let's say 'params' tracks the "mean" model.
        # We update 'params' to minimize VE loss.
        # But we use a sampled model for Q-learning (Thompson Sampling part).
        
        # Update params (Learning)
        grads = dLdp(params)
        # Clip gradients
        grads = (np.clip(grads[0], -1.0, 1.0), np.clip(grads[1], -1.0, 1.0))
        
        new_P_logits = params[0] - lr * grads[0]
        new_r = params[1] - lr * grads[1]
        new_params = (new_P_logits, new_r)
        
        # Sample model for Planning (Symmetric Prior)
        rng_key, subkey = jrandom.split(rng_key)
        m = symmetric_sampler(new_params, subkey)
        
        # Plan on sampled model (One step of Bellman Optimality)
        # Q_ = r + g max Q
        Q_next = utils.bellman_optimality_operator(m.P, m.r, Q, m.discount)
        
        # Soft update Q
        Q_new = Q + lr * (Q_next - Q)
        
        return new_params, Q_new, rng_key

    return update_step, params

if __name__ == "__main__":
    n_states, n_actions = 4, 2
    # Create a true MDP that has some structure (latent MDP)
    # true_mdp = utils.build_random_mdp(n_states, n_actions, 0.9)
    
    # Let's use a simple chain or something
    P = onp.zeros((n_states, n_states, n_actions))
    # ... (construct simple MDP)
    # Just random for now
    true_mdp = utils.build_random_mdp(n_states, n_actions, 0.9)
    
    update_fn, params = thompson(true_mdp)
    
    Q = onp.zeros((n_states, n_actions))
    rng_key = jrandom.PRNGKey(42)
    
    print("Starting training...")
    for i in range(100):
        params, Q, rng_key = update_fn(params, Q, rng_key)
        if i % 10 == 0:
            print(f"Step {i}")
            
    print("Done.")
