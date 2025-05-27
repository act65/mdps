import numpy as np
import pytest
from mdp.utils import (
    value_functional,
    get_deterministic_policies,
    gen_grid_policies,
    random_policy,
    build_random_mdp,
    MDP # Assuming MDP class is in utils or accessible
)

# Test for value_functional
def test_value_functional():
    # Test Case 1
    P1 = np.array([
        [[0.5, 0.5], [0.8, 0.2]],  # Transitions from S0: S0A0 -> S0,S1; S0A1 -> S0,S1
        [[0.3, 0.7], [0.1, 0.9]]   # Transitions from S1: S1A0 -> S0,S1; S1A1 -> S0,S1
    ])
    r1 = np.array([
        [1.0, 0.0],  # Rewards for S0A0, S0A1
        [0.0, 2.0]   # Rewards for S1A0, S1A1
    ])
    pi1 = np.array([
        [1.0, 0.0],  # Policy for S0: take A0
        [0.0, 1.0]   # Policy for S1: take A1
    ])
    discount1 = 0.9
    # Expected V: V = (I - gamma * P_pi)^-1 * r_pi
    # P_pi = [[P(S0'|S0,pi(S0)), P(S1'|S0,pi(S0))], [P(S0'|S1,pi(S1)), P(S1'|S1,pi(S1))]]
    # P_pi1 = [[P(s'|s0,a0)], [P(s'|s1,a1)]]
    # P_pi1_s_s' = [[P(s0|s0,a0), P(s1|s0,a0)], [P(s0|s1,a1), P(s1|s1,a1)]]
    # P_pi1_s_s' = [[0.5, 0.5], [0.1, 0.9]]
    # r_pi1 = [r(s0,a0), r(s1,a1)] = [1.0, 2.0]
    P_pi1_manual = np.array([
        [P1[0,0,0]*pi1[0,0] + P1[1,0,0]*pi1[0,1], P1[0,0,1]*pi1[0,0] + P1[1,0,1]*pi1[0,1]], # This is wrong. P_pi[s, s'] = sum_a pi(a|s) * P(s'|s,a)
        [P1[0,1,0]*pi1[1,0] + P1[1,1,0]*pi1[1,1], P1[0,1,1]*pi1[1,0] + P1[1,1,1]*pi1[1,1]]
    ])
    # Correct P_pi construction: P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a)
    # For pi1:
    # P_pi1[0,0] = pi1[0,0]*P1[0,0,0] + pi1[0,1]*P1[0,0,1] -> This is P(S0'|S0,pi(S0)) - incorrect
    # P_pi[s,s'] = sum_a pi(a|s) * P(s'|s,a)
    # P_pi_s_s_prime = [[P(S0|S0, pi(S0)), P(S1|S0, pi(S0))];
    #                   [P(S0|S1, pi(S1)), P(S1|S1, pi(S1))]]
    # P_pi_s_s_prime = P_pi_correct[s, s'] = sum_a pi(a|s) * P[s',s,a] (because of P definition in code P[next_state, current_state, action])
    # P_pi_s_s_prime_T[s',s] = sum_a pi(a|s) * P[s',s,a]
    # P_pi_T for the function:
    # P_pi_T[s_prime, s] = P[s_prime, s, pi_action_for_s]
    P_pi_T1 = np.array([
        [P1[0,0,0], P1[0,1,1]], # P(S0'|S0,A0), P(S0'|S1,A1)
        [P1[1,0,0], P1[1,1,1]]  # P(S1'|S0,A0), P(S1'|S1,A1)
    ])
    # r_pi1 = [r(S0,A0), r(S1,A1)]
    r_pi1 = np.array([r1[0,0], r1[1,1]]).reshape(-1,1)
    # V = (I - discount * P_pi_T)^-1 * r_pi
    # P_pi_T is from the perspective of the function: P_pi_T[next_state, current_state]
    # P_pi_T_manual = [[0.5, 0.1], [0.5, 0.9]] (P(s'|s,pi(s)))
    # The function uses P_pi.T where P_pi[i,j] is transition prob from state i to state j under policy pi.
    # So P_pi_func_T = P_pi_manual.T
    # P_pi_manual[s,s'] = sum_a pi(a|s) P(s'|s,a)
    # P_pi_manual for pi1:
    # P_pi_manual[0,0] = 1.0 * P1[0,0,0] + 0.0 * P1[0,0,1] = P1[0,0,0] -> P(S0'|S0,A0) This is P(next_state | current_state, action_from_policy)
    # P1 is P[next_state, current_state, action]
    # P_pi_manual[s,s'] = sum_a pi[s,a] * P[s', s, a]
    P_pi_calc = np.zeros((2,2))
    for s_curr in range(2):
        for s_next in range(2):
            for a in range(2):
                P_pi_calc[s_curr, s_next] += pi1[s_curr, a] * P1[s_next, s_curr, a]

    # P_pi_calc = [[0.5, 0.5],  # Transitions from S0 under pi1 (A0): S0->S0, S0->S1
    #              [0.1, 0.9]]  # Transitions from S1 under pi1 (A1): S1->S0, S1->S1
    r_pi_calc = np.zeros((2,1))
    for s in range(2):
        for a in range(2):
            r_pi_calc[s] += pi1[s,a] * r1[s,a]
    # r_pi_calc = [[1.0], [2.0]]

    V_expected1 = np.linalg.inv(np.eye(2) - discount1 * P_pi_calc.T) @ r_pi_calc
    assert np.allclose(value_functional(P1, r1, pi1, discount1), V_expected1)

    # Test Case 2
    P2 = np.array([
        [[0.7, 0.3], [0.4, 0.6]],
        [[0.9, 0.1], [0.2, 0.8]]
    ])
    r2 = np.array([
        [10.0, 5.0],
        [2.0, -1.0]
    ])
    pi2 = np.array([ # Policy: always take action 0
        [1.0, 0.0],
        [1.0, 0.0]
    ])
    discount2 = 0.5
    P_pi_calc2 = np.zeros((2,2))
    for s_curr in range(2):
        for s_next in range(2):
            for a in range(2):
                P_pi_calc2[s_curr, s_next] += pi2[s_curr, a] * P2[s_next, s_curr, a]
    # P_pi_calc2 = [[0.7, 0.3], [0.9, 0.1]]
    r_pi_calc2 = np.zeros((2,1))
    for s in range(2):
        for a in range(2):
            r_pi_calc2[s] += pi2[s,a] * r2[s,a]
    # r_pi_calc2 = [[10.0], [2.0]]
    V_expected2 = np.linalg.inv(np.eye(2) - discount2 * P_pi_calc2.T) @ r_pi_calc2
    assert np.allclose(value_functional(P2, r2, pi2, discount2), V_expected2)


# Test for get_deterministic_policies
def test_get_deterministic_policies():
    n_states, n_actions = 2, 2
    policies = get_deterministic_policies(n_states, n_actions)
    assert len(policies) == n_actions ** n_states  # 2**2 = 4

    for policy in policies:
        assert policy.shape == (n_states, n_actions)
        for s in range(n_states):
            assert np.isclose(np.sum(policy[s, :]), 1.0)
            assert np.sum(policy[s, :] == 1) == 1 # Exactly one 1
            assert np.sum(policy[s, :] == 0) == n_actions - 1 # Others are 0


# Test for gen_grid_policies
def test_gen_grid_policies():
    N_test = 2  # Generates policies for p=0 and p=1
    # For 2 states, 2 actions, this means:
    # p0 for S0, p0 for S1 -> [[0,1],[0,1]]
    # p0 for S0, p1 for S1 -> [[0,1],[1,0]]
    # p1 for S0, p0 for S1 -> [[1,0],[0,1]]
    # p1 for S0, p1 for S1 -> [[1,0],[1,0]]
    # Note: The function's internal logic might lead to a different order or interpretation of "grid"
    # The current implementation iterates prob for s0, then prob for s1.
    # prob = [0,1] for N=2
    # s0_probs = [ [1,0], [0,1] ] (prob for action 0 in state 0 is 1, then 0)
    # s1_probs = [ [1,0], [0,1] ] (prob for action 0 in state 1 is 1, then 0)
    # Policy construction: policy[0,:] = s0_p; policy[1,:] = s1_p
    # Expected:
    # p0=1 for s0a0, p0=1 for s1a0 -> [[1,0],[1,0]]
    # p0=1 for s0a0, p1=1 for s1a0 -> [[1,0],[0,1]] (s1 takes action 1)
    # p1=1 for s0a0, p0=1 for s1a0 -> [[0,1],[1,0]] (s0 takes action 1)
    # p1=1 for s0a0, p1=1 for s1a0 -> [[0,1],[0,1]] (s0 takes action 1, s1 takes action 1)

    policies = gen_grid_policies(N_test)
    assert len(policies) == N_test ** 2 # For 2-state, 2-action, N^S policies

    # Convert list of arrays to a set of tuples for easier comparison
    # policy_set = {tuple(map(tuple, p)) for p in policies}
    # For N=2, probs are [0,1]. Action 0 gets prob, action 1 gets 1-prob
    # s0_ps = [np.array([1.,0.]), np.array([0.,1.])]
    # s1_ps = [np.array([1.,0.]), np.array([0.,1.])]
    expected_policies_list = [
        np.array([[1.0, 0.0], [1.0, 0.0]]), # s0_ps[0], s1_ps[0]
        np.array([[1.0, 0.0], [0.0, 1.0]]), # s0_ps[0], s1_ps[1]
        np.array([[0.0, 1.0], [1.0, 0.0]]), # s0_ps[1], s1_ps[0]
        np.array([[0.0, 1.0], [0.0, 1.0]]), # s0_ps[1], s1_ps[1]
    ]
    
    found_count = 0
    for expected_p in expected_policies_list:
        for p in policies:
            if np.allclose(p, expected_p):
                found_count +=1
                break
    assert found_count == len(expected_policies_list)


# Test for random_policy
def test_random_policy():
    n_states, n_actions = 3, 2
    policy = random_policy(n_states, n_actions)

    assert policy.shape == (n_states, n_actions)
    for s in range(n_states):
        assert np.allclose(np.sum(policy[s, :]), 1.0)
        assert np.all(policy[s,:] >= 0) # Probabilities should be non-negative

# Test for build_random_mdp
def test_build_random_mdp():
    n_states, n_actions, discount = 2, 2, 0.9
    mdp = build_random_mdp(n_states, n_actions, discount)

    assert isinstance(mdp, MDP)
    assert mdp.S == n_states
    assert mdp.A == n_actions
    assert mdp.discount == discount

    assert mdp.P.shape == (n_states, n_states, n_actions) # P[next_state, current_state, action]
    # Probabilities P(s'|s,a) must sum to 1 over s' for each (s,a)
    # So, sum mdp.P along axis 0 (next_state)
    assert np.allclose(np.sum(mdp.P, axis=0), np.ones((n_states, n_actions)))

    assert mdp.r.shape == (n_states, n_actions)
    assert mdp.d0.shape == (n_states, 1)
    assert np.isclose(np.sum(mdp.d0), 1.0)

```
