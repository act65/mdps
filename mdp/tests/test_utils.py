import pytest
import jax.numpy as jnp
from numpy.typing import NDArray # For type hinting with NumPy if needed, but JAX is primary
import numpy.random as rnd # For generating random numbers (standard NumPy)
import numpy as np # For np.testing and some array creations for expected values.
from typing import List, Tuple, Any, Callable # For type hints

from mdp.utils import (
    MDP, # Now a typing.NamedTuple
    onehot,
    entropy,
    sigmoid,
    softmax,
    normalize,
    clip_by_norm,
    build_random_mdp,
    build_random_sparse_mdp,
    sparsify,
    gen_grid_policies,
    get_deterministic_policies,
    get_random_policy_2x2,
    rnd_simplex,
    random_policy,
    random_det_policy,
    polytope,
    value_functional,
    bellman_optimality_operator,
    bellman_operator,
    isclose,
    converged,
    solve,
    discounted_rewards,
    sample_from_distribution, # Renamed from sample
    rollout
)
import mdp.search_spaces as search_spaces # For isclose testing

# --- Tests for Basic Utility Functions ---

def test_onehot():
    assert jnp.array_equal(onehot(0, 3), jnp.array([1., 0., 0.]))
    assert jnp.array_equal(onehot(1, 3), jnp.array([0., 1., 0.]))
    assert jnp.array_equal(onehot(2, 3), jnp.array([0., 0., 1.]))
    with pytest.raises(IndexError): # Or similar JAX error
        onehot(3,3)

def test_entropy():
    # Uniform distribution: H = log(N)
    assert jnp.isclose(entropy(jnp.array([0.5, 0.5])), jnp.log(2.0))
    assert jnp.isclose(entropy(jnp.array([1/3, 1/3, 1/3])), jnp.log(3.0))
    # Deterministic distribution: H = 0
    assert jnp.isclose(entropy(jnp.array([1.0, 0.0, 0.0])), 0.0, atol=1e-7) # Epsilon in func affects this
    # Test with epsilon for stability
    assert jnp.isclose(entropy(jnp.array([1.0, 1e-9, 1e-9])), 0.0, atol=1e-5)


def test_sigmoid():
    assert jnp.isclose(sigmoid(jnp.array(0.0)), jnp.array(0.5))
    assert sigmoid(jnp.array([-jnp.inf, 0, jnp.inf]))[0] < 1e-6 # approx 0
    assert jnp.isclose(sigmoid(jnp.array([-jnp.inf, 0, jnp.inf]))[1], 0.5)
    assert sigmoid(jnp.array([-jnp.inf, 0, jnp.inf]))[2] > 1.0 - 1e-6 # approx 1

def test_softmax():
    x = jnp.array([1.0, 2.0, 3.0])
    sm_x = softmax(x)
    np.testing.assert_allclose(jnp.sum(sm_x), 1.0)
    assert sm_x[0] < sm_x[1] < sm_x[2]

    x_2d = jnp.array([[1.,2.],[0.,1.]])
    sm_x_axis0 = softmax(x_2d, axis=0)
    np.testing.assert_allclose(jnp.sum(sm_x_axis0, axis=0), jnp.array([1.0, 1.0]))
    sm_x_axis1 = softmax(x_2d, axis=1)
    np.testing.assert_allclose(jnp.sum(sm_x_axis1, axis=1), jnp.array([1.0, 1.0]))


def test_normalize():
    x = jnp.array([[3.0, 4.0], [0.0, 0.0]]) # Second vector has norm 0
    norm_x = normalize(x, axis=1, epsilon=1e-8) # Normalize rows
    np.testing.assert_allclose(jnp.linalg.norm(norm_x[0]), 1.0)
    np.testing.assert_allclose(jnp.linalg.norm(norm_x[1]), 0.0) # Norm of zero vector is zero
    np.testing.assert_allclose(norm_x[0], jnp.array([0.6, 0.8]))

def test_clip_by_norm():
    vec = jnp.array([1.0, 2.0, 2.0]) # Norm is 3
    clipped = clip_by_norm(vec, max_norm=1.5)
    np.testing.assert_allclose(jnp.linalg.norm(clipped), 1.5)
    np.testing.assert_allclose(clipped, vec * 0.5)

    clipped_no_change = clip_by_norm(vec, max_norm=4.0)
    np.testing.assert_allclose(clipped_no_change, vec)

# --- Tests for MDP Construction ---
def test_build_random_mdp():
    num_states, num_actions, discount = 3, 2, 0.95
    mdp = build_random_mdp(num_states, num_actions, discount)

    assert isinstance(mdp, MDP)
    assert mdp.S == num_states
    assert mdp.A == num_actions
    assert mdp.discount == discount

    assert mdp.P.shape == (num_states, num_states, num_actions) # P[s_next, s_current, action]
    # Probabilities P(s'|s,a) must sum to 1 over s' (axis 0) for each (s,a)
    np.testing.assert_allclose(jnp.sum(mdp.P, axis=0), jnp.ones((num_states, num_actions)), atol=1e-6)

    assert mdp.r.shape == (num_states, num_actions) # r[s_current, action]
    assert mdp.d0.shape == (num_states, 1)
    np.testing.assert_allclose(jnp.sum(mdp.d0), 1.0, atol=1e-6)

def test_sparsify():
    arr_np = rnd.rand(10,10)
    arr_jnp = jnp.array(arr_np)
    sparsified_arr = sparsify(arr_jnp, sparsity_factor=0.8) # ~80% zeros
    assert jnp.sum(sparsified_arr == 0) >= 0.7 * arr_jnp.size # Check for significant number of zeros

def test_build_random_sparse_mdp():
    num_states, num_actions, discount = 4, 3, 0.8
    mdp = build_random_sparse_mdp(num_states, num_actions, discount, sparsity=0.7)
    assert mdp.P.shape == (num_states, num_states, num_actions)
    np.testing.assert_allclose(jnp.sum(mdp.P, axis=0), jnp.ones((num_states, num_actions)), atol=1e-6)
    # Check if P and r are actually sparse (contains zeros)
    assert jnp.any(mdp.P == 0), "Sparse P should contain zeros"
    assert jnp.any(mdp.r == 0), "Sparse r should contain zeros"

# --- Tests for Policy Generation ---
def test_gen_grid_policies():
    # Test the S=2, A=2 case
    num_points = 2 # Generates policies for p=0 and p=1
    policies = gen_grid_policies(num_points_per_dim=num_points, num_states=2, num_actions=2)
    assert len(policies) == num_points ** 2 

    expected_policies_set = {
        tuple(map(tuple, np.array([[1.0, 0.0], [1.0, 0.0]]))),
        tuple(map(tuple, np.array([[1.0, 0.0], [0.0, 1.0]]))),
        tuple(map(tuple, np.array([[0.0, 1.0], [1.0, 0.0]]))),
        tuple(map(tuple, np.array([[0.0, 1.0], [0.0, 1.0]]))),
    }
    generated_policies_set = {tuple(map(tuple, p.tolist())) for p in policies} # Convert JAX to list for frozenset
    assert generated_policies_set == expected_policies_set
    
    with pytest.raises(NotImplementedError):
        gen_grid_policies(num_points, num_states=3, num_actions=2)


def test_get_deterministic_policies():
    num_states, num_actions = 2, 2
    policies = get_deterministic_policies(num_states, num_actions)
    assert len(policies) == num_actions ** num_states  # 2^2 = 4

    for policy in policies:
        assert policy.shape == (num_states, num_actions)
        for s_idx in range(num_states):
            np.testing.assert_allclose(jnp.sum(policy[s_idx, :]), 1.0)
            assert jnp.sum(policy[s_idx, :] == 1) == 1 # Exactly one 1
            assert jnp.sum(policy[s_idx, :] == 0) == num_actions - 1 # Others are 0

def test_get_random_policy_2x2():
    policy = get_random_policy_2x2()
    assert policy.shape == (2,2)
    np.testing.assert_allclose(jnp.sum(policy, axis=1), jnp.array([1.0, 1.0]))

def test_rnd_simplex():
    for dim in [1, 2, 5, 10]:
        simplex_point = rnd_simplex(dim)
        assert simplex_point.shape == (dim,)
        np.testing.assert_allclose(jnp.sum(simplex_point), 1.0, atol=1e-6)
        assert jnp.all(simplex_point >= 0)

def test_random_policy():
    num_states, num_actions = 3, 4
    policy = random_policy(num_states, num_actions)
    assert policy.shape == (num_states, num_actions)
    np.testing.assert_allclose(jnp.sum(policy, axis=1), jnp.ones(num_states), atol=1e-6)

def test_random_det_policy():
    num_states, num_actions = 3, 4
    policy = random_det_policy(num_states, num_actions)
    assert policy.shape == (num_states, num_actions)
    for s_idx in range(num_states):
        np.testing.assert_allclose(jnp.sum(policy[s_idx, :]), 1.0)
        assert jnp.sum(policy[s_idx, :] == 1) == 1

# --- Tests for Value Function and Bellman Operators ---
def test_polytope_simple():
    num_states, num_actions, discount = 2,1,0.9
    P = jnp.array([[[1.0],[0.0]], [[0.0],[1.0]]]) # P[s',s,a]: S0->S0, S1->S1
    r = jnp.array([[1.0],[2.0]])
    
    # Policy 1: A0 in S0, A0 in S1
    pi1 = jnp.array([[1.0],[1.0]]) 
    # V_pi1: V0 = 1 + 0.9*V0 => V0=10. V1 = 2 + 0.9*V1 => V1=20.  V_pi1 = [10,20]
    
    policies = [pi1]
    value_polytope = polytope(P,r,discount,policies)
    assert value_polytope.shape == (len(policies), num_states)
    np.testing.assert_allclose(value_polytope[0], jnp.array([10.0, 20.0]), atol=1e-5)


def test_value_functional_simple_mdp():
    # MDP: 2 states (S0, S1), 1 action (A0)
    # P(S0|S0,A0)=1, P(S1|S1,A0)=1 (deterministic self-loops)
    # r(S0,A0)=1, r(S1,A0)=2
    # pi(A0|S0)=1, pi(A0|S1)=1
    # V(S0) = 1 + 0.9*V(S0) => 0.1V(S0)=1 => V(S0)=10
    # V(S1) = 2 + 0.9*V(S1) => 0.1V(S1)=2 => V(S1)=20
    P = jnp.array([[[1.0]], [[0.0]]]) # P[s_next=0, s_current=0, action=0]
    P = jnp.array([ # P[s_next, s_current, action]
        [[1.0], [0.0]], # Transitions to S0 from (S0,A0), (S1,A0)
        [[0.0], [1.0]]  # Transitions to S1 from (S0,A0), (S1,A0)
    ])
    r = jnp.array([[1.0], [2.0]]) # r[s_current, action]
    pi = jnp.array([[1.0], [1.0]]) # pi[s_current, action]
    discount = 0.9
    
    V_expected = jnp.array([[10.0], [20.0]])
    V_calculated = value_functional(P, r, pi, discount)
    np.testing.assert_allclose(V_calculated, V_expected, atol=1e-5)

def test_bellman_optimality_operator_simple():
    P = jnp.array([[[0.5,0.5],[0.8,0.2]], [[0.3,0.7],[0.1,0.9]]]) # P[s',s,a]
    r = jnp.array([[1.,0.],[0.,2.]]) # r[s,a]
    Q_curr = jnp.array([[10.,8.],[20.,22.]]) # Q[s,a]
    discount = 0.9
    # V_max_s_prime = [max(10,8), max(20,22)] = [10, 22]
    # EV_s0_a0 = P[0,0,0]*V_max[0] + P[1,0,0]*V_max[1] = 0.5*10 + 0.3*22 = 5 + 6.6 = 11.6
    # EV_s0_a1 = P[0,0,1]*V_max[0] + P[1,0,1]*V_max[1] = 0.8*10 + 0.1*22 = 8 + 2.2 = 10.2
    # EV_s1_a0 = P[0,1,0]*V_max[0] + P[1,1,0]*V_max[1] = 0.5*10 + 0.7*22 = 5 + 15.4 = 20.4
    # EV_s1_a1 = P[0,1,1]*V_max[0] + P[1,1,1]*V_max[1] = 0.2*10 + 0.9*22 = 2 + 19.8 = 21.8
    # Q_next[0,0] = r[0,0] + disc * EV_s0_a0 = 1 + 0.9 * 11.6 = 1 + 10.44 = 11.44
    # Q_next[0,1] = r[0,1] + disc * EV_s0_a1 = 0 + 0.9 * 10.2 = 9.18
    # Q_next[1,0] = r[1,0] + disc * EV_s1_a0 = 0 + 0.9 * 20.4 = 18.36
    # Q_next[1,1] = r[1,1] + disc * EV_s1_a1 = 2 + 0.9 * 21.8 = 2 + 19.62 = 21.62
    Q_expected = jnp.array([[11.44, 9.18], [18.36, 21.62]])
    Q_next = bellman_optimality_operator(P,r,Q_curr,discount)
    np.testing.assert_allclose(Q_next, Q_expected, atol=1e-4)


def test_bellman_operator_simple():
    P = jnp.array([[[0.5,0.5],[0.8,0.2]], [[0.3,0.7],[0.1,0.9]]]) # P[s',s,a]
    r = jnp.array([[1.,0.],[0.,2.]]) # r[s,a]
    V_s_prime = jnp.array([10.,20.]) # V[s']
    discount = 0.9
    # EV_s0_a0 = P[0,0,0]*V[0] + P[1,0,0]*V[1] = 0.5*10 + 0.3*20 = 5 + 6 = 11
    # EV_s0_a1 = P[0,0,1]*V[0] + P[1,0,1]*V[1] = 0.8*10 + 0.1*20 = 8 + 2 = 10
    # EV_s1_a0 = P[0,1,0]*V[0] + P[1,1,0]*V[1] = 0.5*10 + 0.7*20 = 5 + 14 = 19
    # EV_s1_a1 = P[0,1,1]*V[0] + P[1,1,1]*V[1] = 0.2*10 + 0.9*20 = 2 + 18 = 20
    # Q_next[0,0] = r[0,0] + disc * EV_s0_a0 = 1 + 0.9 * 11 = 1 + 9.9 = 10.9
    # Q_next[0,1] = r[0,1] + disc * EV_s0_a1 = 0 + 0.9 * 10 = 9.0
    # Q_next[1,0] = r[1,0] + disc * EV_s1_a0 = 0 + 0.9 * 19 = 17.1
    # Q_next[1,1] = r[1,1] + disc * EV_s1_a1 = 2 + 0.9 * 20 = 2 + 18 = 20.0
    Q_expected = jnp.array([[10.9, 9.0], [17.1, 20.0]])
    Q_next = bellman_operator(P,r,V_s_prime,discount)
    np.testing.assert_allclose(Q_next, Q_expected, atol=1e-4)

# --- Tests for Iterative Solving ---
def test_isclose():
    assert isclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.000000001])) == True
    assert isclose(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.00001])) == False
    assert isclose([jnp.array([1.])], [jnp.array([1.000000001])]) == True # List of arrays
    assert isclose((jnp.array([1.]),), (jnp.array([1.000000001]),)) == True # Tuple of arrays
    assert isclose(1.0, 1.000000001) == True

def test_converged():
    history1 = [jnp.array([1.0]), jnp.array([1.0])] # Converged immediately
    assert converged(history1, high_tol=1e-7, iteration_threshold_high_tol=1) == True 
    history2 = [jnp.array([1.0]), jnp.array([1.1]), jnp.array([1.01]), jnp.array([1.001])]
    assert converged(history2, high_tol=1e-7, iteration_threshold_high_tol=1) == False
    history3 = [jnp.array([1.0])] * 20001 # Long history, but not changing
    with pytest.raises(ValueError, match="Not converged after"):
         converged(history3, max_iterations_error=20000) # Should raise error
    history4 = [jnp.array([1.0])] * 10 + [jnp.array([1.0000000001])] * 10
    assert converged(history4, high_tol=1e-8, iteration_threshold_high_tol=1, max_iterations_error=30) == True

def test_solve_simple_fixed_point():
    # Test x_next = x/2 + 1. Fixed point is x = x/2 + 1 => x/2 = 1 => x = 2.
    update_fn = lambda x: x / 2.0 + 1.0
    initial_val = jnp.array(0.0)
    history = solve(update_fn, initial_val, tol=1e-7, max_iter=100) # solve needs max_iter and tol
    assert jnp.isclose(history[-1], jnp.array(2.0), atol=1e-6)

    # Test with tuple state (e.g. for momentum)
    # update_fn( (x,y) ) -> (new_x, new_y)
    # Example: x_t = 0.9*x_{t-1} + 0.1, y_t = 0.5*y_{t-1}
    # Converges to x=1, y=0
    def update_tuple_fn(state_tuple):
        x, y = state_tuple
        new_x = 0.9 * x + 0.1
        new_y = 0.5 * y
        return (new_x, new_y)
    initial_tuple_state = (jnp.array(0.0), jnp.array(1.0))
    history_tuple = solve(update_tuple_fn, initial_tuple_state, tol=1e-7, max_iter=200)
    final_x, final_y = history_tuple[-1]
    assert jnp.isclose(final_x, 1.0, atol=1e-6)
    assert jnp.isclose(final_y, 0.0, atol=1e-6)


# --- Tests for Simulation and Sampling ---
def test_discounted_rewards():
    rewards = jnp.array([1.0, 1.0, 1.0])
    discount = 0.9
    # (1-0.9) * (1*0.9^0 + 1*0.9^1 + 1*0.9^2)
    # = 0.1 * (1 + 0.9 + 0.81) = 0.1 * 2.71 = 0.271
    expected_dr = (1-discount) * (1*discount**0 + 1*discount**1 + 1*discount**2)
    assert jnp.isclose(discounted_rewards(rewards, discount), expected_dr)

def test_sample_from_distribution():
    # Test with a distribution peaked at one index
    probs = jnp.array([0.01, 0.01, 0.95, 0.01, 0.02]) # Peaks at index 2
    # With low temperature, should mostly pick index 2
    samples = [sample_from_distribution(probs, temperature=0.001).item() for _ in range(100)]
    assert sum(s == 2 for s in samples) > 70 # High probability of picking mode

    # Test that output is a valid index
    for _ in range(10):
        idx = sample_from_distribution(probs, temperature=1.0).item()
        assert 0 <= idx < len(probs)
    
    # Test deterministic case (temp=0)
    assert sample_from_distribution(probs, temperature=0.0).item() == 2


def test_rollout_deterministic_mdp():
    # MDP: S0 --A0--> S1 --A0--> S0 (cycle)
    # r(S0,A0)=1, r(S1,A0)=2
    num_states, num_actions = 2, 1
    P = jnp.zeros((num_states, num_states, num_actions)) # P[s_next, s_current, action]
    P = P.at[1, 0, 0].set(1.0)  # S0, A0 -> S1
    P = P.at[0, 1, 0].set(1.0)  # S1, A0 -> S0
    
    r = jnp.array([[1.0], [2.0]]) # r[s_current, action]
    
    d0 = jnp.array([1.0, 0.0]) # Start in S0
    policy = jnp.array([[1.0], [1.0]]) # Always take A0 (only action)
    num_timesteps = 4

    trajectory = rollout(P, r, d0, policy, num_timesteps)
    
    assert len(trajectory) == num_timesteps
    # Expected: (S0,A0,R1) -> (S1,A0,R2) -> (S0,A0,R1) -> (S1,A0,R2)
    expected_trajectory = [
        (0, 0, 1.0),
        (1, 0, 2.0),
        (0, 0, 1.0),
        (1, 0, 2.0),
    ]
    assert trajectory == expected_trajectory