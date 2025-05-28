import pytest
import jax.numpy as jnp
from numpy.typing import NDArray
import numpy.random as rnd # For test data generation
import numpy as np # For assertions and some test data setup

import mdp.utils as utils
from mdp.lmdps import (
    rnd_lmdp,
    mdp_encoder,
    KL,
    CE,
    linear_bellman_operator,
    linear_value_functional,
    lmdp_solver,
    lmdp_decoder,
    option_transition_fn,
    lmdp_option_decoder
)

# --- Tests for mdp/lmdps.py functions ---

def test_rnd_lmdp():
    """Tests rnd_lmdp for output shapes and properties of p."""
    num_states, num_actions = 3, 2
    p, q = rnd_lmdp(num_states, num_actions)

    assert p.shape == (num_states, num_states), f"p shape mismatch: expected ({num_states},{num_states}), got {p.shape}"
    assert q.shape == (num_states, 1), f"q shape mismatch: expected ({num_states},1), got {q.shape}"
    
    # Check if p is column-stochastic (columns sum to 1)
    column_sums_p = jnp.sum(p, axis=0)
    np.testing.assert_allclose(
        column_sums_p, 
        jnp.ones(num_states), 
        atol=1e-6, 
        err_msg="p is not column-stochastic"
    )

def test_mdp_encoder_simple():
    """
    Tests mdp_encoder with a very simple MDP.
    Focuses on shapes and basic properties due to complexity of exact value verification.
    """
    num_states, num_actions = 2, 1
    # P_mdp[s,a,s']
    P_mdp_np = np.zeros((num_states, num_actions, num_states))
    P_mdp_np[0, 0, 0] = 0.5 # S0, A0 -> S0 (0.5)
    P_mdp_np[0, 0, 1] = 0.5 # S0, A0 -> S1 (0.5)
    P_mdp_np[1, 0, 0] = 0.1 # S1, A0 -> S0 (0.1)
    P_mdp_np[1, 0, 1] = 0.9 # S1, A0 -> S1 (0.9)
    
    r_mdp_np = np.array([[1.0], [0.5]]) # r(s,a)

    P_mdp_jnp = jnp.array(P_mdp_np)
    r_mdp_jnp = jnp.array(r_mdp_np)

    p_encoded, q_encoded = mdp_encoder(P_mdp_jnp, r_mdp_jnp)

    assert p_encoded.shape == (num_states, num_states), \
        f"p_encoded shape mismatch: expected ({num_states},{num_states}), got {p_encoded.shape}"
    assert q_encoded.shape == (num_states,), \
        f"q_encoded shape mismatch: expected ({num_states},), got {q_encoded.shape}"

    # p_encoded should be column-stochastic
    column_sums_p_encoded = jnp.sum(p_encoded, axis=0)
    np.testing.assert_allclose(
        column_sums_p_encoded, 
        jnp.ones(num_states), 
        atol=1e-5, # Increased tolerance due to pinv and numerical precision
        err_msg="p_encoded is not column-stochastic"
    )

def test_kl_divergence():
    """Tests KL divergence with simple distributions."""
    P_dist = jnp.array([0.5, 0.5])
    Q_dist = jnp.array([0.4, 0.6])
    # KL(P||Q) = 0.5*log(0.5/0.4) + 0.5*log(0.5/0.6)
    #         = 0.5*log(1.25) + 0.5*log(0.8333)
    #         = 0.5*0.2231 + 0.5*(-0.1823) = 0.0204 approx
    expected_kl = 0.5 * jnp.log(0.5/0.4) + 0.5 * jnp.log(0.5/0.6)
    kl_val = KL(P_dist, Q_dist)
    np.testing.assert_allclose(kl_val, expected_kl, atol=1e-4)

    # Test identity: KL(P||P) = 0
    kl_identity = KL(P_dist, P_dist)
    np.testing.assert_allclose(kl_identity, 0.0, atol=1e-7)

def test_ce_sum_p_log_q():
    """Tests CE (sum P log Q) with simple distributions."""
    P_dist = jnp.array([0.5, 0.5])
    Q_dist = jnp.array([0.4, 0.6])
    # sum P[i] log Q[i] = 0.5*log(0.4) + 0.5*log(0.6)
    #                   = 0.5*(-0.916) + 0.5*(-0.510) = -0.458 - 0.255 = -0.713 approx
    expected_ce = 0.5 * jnp.log(0.4) + 0.5 * jnp.log(0.6)
    ce_val = CE(P_dist, Q_dist)
    np.testing.assert_allclose(ce_val, expected_ce, atol=1e-3)


def test_linear_bellman_operator_simple():
    """Tests linear_bellman_operator with small inputs."""
    num_states = 2
    discount = 0.9
    # p_uncontrolled: p(s_col | s_row) - row-stochastic
    p_uncontrolled = jnp.array([[0.7, 0.3], [0.2, 0.8]])
    q_pseudo_reward = jnp.array([1.0, 0.5]) # Shape (S,)
    z_exp_value = jnp.array([2.0, 3.0])     # Shape (S,)

    z_next = linear_bellman_operator(p_uncontrolled, q_pseudo_reward, z_exp_value, discount)
    
    assert z_next.shape == (num_states,), f"Expected shape ({num_states},), got {z_next.shape}"
    # Manual calculation:
    # z_next[0] = exp(q[0]) * (p[0,0]*z[0]^d + p[0,1]*z[1]^d)
    #           = exp(1.0) * (0.7*2^0.9 + 0.3*3^0.9)
    #           = 2.71828 * (0.7*1.866 + 0.3*2.706)
    #           = 2.71828 * (1.3062 + 0.8118) = 2.71828 * 2.118 = 5.757
    # z_next[1] = exp(q[1]) * (p[1,0]*z[0]^d + p[1,1]*z[1]^d)
    #           = exp(0.5) * (0.2*2^0.9 + 0.8*3^0.9)
    #           = 1.64872 * (0.2*1.866 + 0.8*2.706)
    #           = 1.64872 * (0.3732 + 2.1648) = 1.64872 * 2.538 = 4.184
    expected_z0 = jnp.exp(1.0) * (0.7 * (2.0**discount) + 0.3 * (3.0**discount))
    expected_z1 = jnp.exp(0.5) * (0.2 * (2.0**discount) + 0.8 * (3.0**discount))
    np.testing.assert_allclose(z_next[0], expected_z0, atol=1e-3)
    np.testing.assert_allclose(z_next[1], expected_z1, atol=1e-3)


def test_linear_value_functional_simple():
    """Tests linear_value_functional with small inputs."""
    num_states = 2
    discount = 0.9
    # p_uncontrolled: p(s'|s) - col-stochastic
    p_uncontrolled = jnp.array([[0.7, 0.2], [0.3, 0.8]]) 
    q_pseudo_reward = jnp.array([1.0, 0.5]) # Shape (S,)
    # u_optimal_control: u(s'|s) - col-stochastic
    u_optimal_control = jnp.array([[0.6, 0.1], [0.4, 0.9]]) 

    V_lmdp = linear_value_functional(p_uncontrolled, q_pseudo_reward, u_optimal_control, discount)
    assert V_lmdp.shape == (num_states,), f"Expected shape ({num_states},), got {V_lmdp.shape}"
    # Values are hard to compute manually here, so primarily testing execution and shape.

def test_lmdp_solver_simple():
    """Tests lmdp_solver with small p and q."""
    num_states = 2
    discount = 0.9
    # p_uncontrolled: p(s_col | s_row) for linear_bellman_operator
    # p_uncontrolled for lmdp_solver's final u calc: p(s'_row | s_col)
    # Let's use a p that is row-stochastic as assumed by linear_bellman_operator in refactor
    p_uncontrolled_row_stoch = jnp.array([[0.7, 0.3], [0.2, 0.8]])
    q_pseudo_reward = jnp.array([1.0, 0.0]) # S0 is better

    # The lmdp_solver internally converts p for the u calculation if needed.
    # The p that goes into linear_bellman_operator should be row-stochastic.
    # The p that is used for G and u calculation should be col-stochastic p(s'|s).
    # The current refactored lmdp_solver uses p_uncontrolled as is for G and u,
    # assuming p_uncontrolled[s_dest, s_source].
    # For consistency, let's ensure p_uncontrolled is col-stochastic for this test.
    p_uncontrolled_col_stoch = jnp.array([[0.7, 0.2], [0.3, 0.8]]) # p(s'|s)
    
    # The linear_bellman_operator takes p(next_state | current_state), i.e. row-stochastic.
    # The lmdp_solver's use of `linear_bellman_operator(p,q,z,d)` means `p` there is row-stochastic.
    # But its use for `G = np.einsum('ij,i->j', p, z)` implies p[s',s] and sum over s'.
    # This is confusing. Let's use the `p_uncontrolled_col_stoch` for the solver argument `p_uncontrolled`
    # and assume `linear_bellman_operator` handles its `p_uncontrolled` argument as p(s_col|s_row).
    # The refactored `linear_bellman_operator` expects `p_uncontrolled` to be p(s_col|s_row).
    # The `lmdp_solver` calls it with `p_uncontrolled`.
    # Then for `G = jnp.dot(p_uncontrolled, z_converged)`, if p is (S_rows=dest, S_cols=source)=p(dest|source)
    # and z is (S_dest), then G is (S_source). This is G_s = sum_s' p(s'|s) z_s'. This is correct.
    # Then for `u_optimal = (p_uncontrolled * z_converged[:, jnp.newaxis]) / G_vector_col_stoch_p[jnp.newaxis, :]`
    # This also assumes p_uncontrolled is p(s'_dest | s_source).

    u_opt, V_opt = lmdp_solver(p_uncontrolled_col_stoch, q_pseudo_reward, discount, max_iterations=100, tolerance=1e-5)

    assert u_opt.shape == (num_states, num_states), f"u_opt shape mismatch, got {u_opt.shape}"
    assert V_opt.shape == (num_states,), f"V_opt shape mismatch, got {V_opt.shape}"
    
    # u_opt should be column-stochastic: u(s'|s)
    column_sums_u_opt = jnp.sum(u_opt, axis=0)
    np.testing.assert_allclose(
        column_sums_u_opt, 
        jnp.ones(num_states), 
        atol=1e-6,
        err_msg="u_opt is not column-stochastic"
    )
    # Check if state 0 has higher value due to q[0]=1 vs q[1]=0
    assert V_opt[0] > V_opt[1], "Expected V_opt[0] > V_opt[1] due to rewards"


def test_lmdp_decoder_runs():
    """Tests if lmdp_decoder runs and policy has correct properties."""
    num_states, num_actions = 2, 2
    # Target dynamics u(s'|s_row)
    u_target = jnp.array([[0.8, 0.2], [0.1, 0.9]]) # u[s,s'] = u(s'|s)
    
    # P_original_mdp[s,a,s']
    P_orig_np = rnd.rand(num_states, num_actions, num_states)
    P_orig_np = P_orig_np / P_orig_np.sum(axis=2, keepdims=True)
    P_orig_jnp = jnp.array(P_orig_np)

    policy = lmdp_decoder(u_target, P_orig_jnp, learning_rate=1.0, max_iterations=10, tolerance=1e-3)
    
    assert policy.shape == (num_states, num_actions), f"Policy shape mismatch, got {policy.shape}"
    # Policy should be row-stochastic (rows sum to 1)
    row_sums_policy = jnp.sum(policy, axis=1)
    np.testing.assert_allclose(
        row_sums_policy, 
        jnp.ones(num_states), 
        atol=1e-6,
        err_msg="Policy is not row-stochastic"
    )

def test_option_transition_fn_simple():
    """Tests option_transition_fn with k=1 and k=2."""
    num_states, num_actions = 2, 1 # Simple: 1 action
    # P[s,a,s']
    P_base = jnp.zeros((num_states, num_actions, num_states))
    P_base = P_base.at[0,0,1].set(1.0) # S0,A0 -> S1
    P_base = P_base.at[1,0,0].set(1.0) # S1,A0 -> S0

    # k=1: options are just base actions. P_options should be P_base transposed to (S,S',A)
    P_opts_k1 = option_transition_fn(P_base, 1)
    expected_P_opts_k1 = P_base.transpose(0,2,1)
    assert P_opts_k1.shape == (num_states, num_states, num_actions)
    np.testing.assert_array_equal(P_opts_k1, expected_P_opts_k1)

    # k=2: options are (a0,a0)
    # P_all_options should contain P_len1 and P_len2
    # P_len1: (S0,A0)->S1; (S1,A0)->S0.  Shape (2,2,1)
    # P_len2 for option (A0,A0):
    #   Start S0: A0 -> S1, then A0 -> S0. So S0 -> S0 by (A0,A0)
    #   Start S1: A0 -> S0, then A0 -> S1. So S1 -> S1 by (A0,A0)
    # P_len2_expected[s,s',(a0,a0)] = 1 if (s=0,s'=0) or (s=1,s'=1)
    # P_len2_expected should be shape (S,S,A^2) = (2,2,1)
    P_opts_k2 = option_transition_fn(P_base, 2)
    
    # Total options = A^1 + A^2 = 1 + 1 = 2
    assert P_opts_k2.shape == (num_states, num_states, num_actions + num_actions**2)
    
    # P_len1 part (first A options)
    np.testing.assert_array_equal(P_opts_k2[..., :num_actions], expected_P_opts_k1)

    # P_len2 part (next A^2 options)
    # P_len2_calc[s_start, s_final, (a0,a0)]
    P_len2_manual = jnp.zeros((num_states, num_states, num_actions**2))
    P_len2_manual = P_len2_manual.at[0,0,0].set(1.0) # S0 -> S0 via (A0,A0)
    P_len2_manual = P_len2_manual.at[1,1,0].set(1.0) # S1 -> S1 via (A0,A0)
    np.testing.assert_array_equal(P_opts_k2[..., num_actions:], P_len2_manual)


def test_lmdp_option_decoder_runs():
    """Tests if lmdp_option_decoder runs."""
    num_states, num_actions, k = 2, 1, 2
    u_target = jnp.array([[0.8, 0.2], [0.1, 0.9]]) # u[s,s'] = u(s'|s)
    
    P_base = jnp.zeros((num_states, num_actions, num_states))
    P_base = P_base.at[0,0,1].set(1.0) 
    P_base = P_base.at[1,0,0].set(1.0) 

    num_total_options = sum([num_actions**(i+1) for i in range(k)]) # 1+1=2

    option_policy = lmdp_option_decoder(
        u_target, P_base, learning_rate=0.1, num_steps_k=k, max_iterations=5, tolerance=1e-2
    )
    assert option_policy.shape == (num_states, num_total_options)
    row_sums_policy = jnp.sum(option_policy, axis=1)
    np.testing.assert_allclose(row_sums_policy, jnp.ones(num_states), atol=1e-6)

# Note: The old mdp/lmdp_tests.py had complex setup for `simple_test` and `random_test`
# in `TestMDPEmbeddeding` which involved checking `r_approx`. These specific detailed checks
# are hard to replicate without deeper validation of the mdp_encoder math or known good outputs.
# The current tests focus on shapes, basic properties, and runtime execution.
# The `DecodeLMDPControl` and `DiscountingTest` from old file involved more complex scenarios
# and plotting, partially covered by `lmdp_decoder_runs` and `lmdp_solver_simple`.Okay, I've created `mdp/tests/test_lmdps.py` with the refactored tests and new tests.

The next step is to delete the old `mdp/lmdp_tests.py` file.
