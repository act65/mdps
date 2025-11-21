import mdp.utils as utils
from mdp.density import distributional_value_function
import numpy as np
import pytest

def test_distributional_value_function_simple():
    # Simple chain: 0 -> 1 -> 2 (absorbing)
    # Rewards: 0->1: 1, 1->2: 1
    # Discount: 0.5
    
    n_states = 3
    n_actions = 1
    
    P = np.zeros((n_states, n_states, n_actions))
    P[1, 0, 0] = 1.0 # 0 -> 1
    P[2, 1, 0] = 1.0 # 1 -> 2
    P[2, 2, 0] = 1.0 # 2 -> 2 (absorbing)
    
    r = np.zeros((n_states, n_actions))
    r[0, 0] = 1.0
    r[1, 0] = 1.0
    r[2, 0] = 0.0
    
    pi = np.ones((n_states, n_actions))
    discount = 0.5
    
    V, Var = distributional_value_function(P, r, pi, discount)
    
    # Expected Values:
    # V(2) = 0
    # V(1) = 1 + 0.5 * V(2) = 1
    # V(0) = 1 + 0.5 * V(1) = 1.5
    
    assert np.isclose(V[2], 0.0)
    assert np.isclose(V[1], 1.0)
    assert np.isclose(V[0], 1.5)
    
    # Variance:
    # Deterministic system, variance should be 0?
    # No, return is deterministic sum of rewards.
    # Z(2) = 0
    # Z(1) = 1 + 0.5 * 0 = 1 (const) -> Var=0
    # Z(0) = 1 + 0.5 * 1 = 1.5 (const) -> Var=0
    
    assert np.isclose(Var, 0.0, atol=1e-5).all()

def test_distributional_value_function_stochastic():
    # State 0 -> 1 (prob 0.5, r=1) or 2 (prob 0.5, r=-1)
    # 1, 2 absorbing, r=0
    
    n_states = 3
    n_actions = 1
    
    P = np.zeros((n_states, n_states, n_actions))
    P[1, 0, 0] = 0.5
    P[2, 0, 0] = 0.5
    P[1, 1, 0] = 1.0
    P[2, 2, 0] = 1.0
    
    r = np.zeros((n_states, n_actions))
    r[0, 0] = 0.0 # Reward is 0 at step, but let's say transition reward? 
    # The model assumes r(s,a). So immediate reward is deterministic given s,a.
    # Uncertainty comes from next state.
    
    # Let's make next state values different.
    # V(1) = 10, V(2) = -10
    # To achieve this, let's set rewards at 1 and 2 to be 10(1-gamma)/gamma? No absorbing.
    # Let's just say r(1)=10, r(2)=-10 and they transition to themselves with discount.
    # V(1) = 10 + g V(1) -> V(1) = 10/(1-g)
    
    discount = 0.5
    r[1, 0] = 5.0 # V(1) = 5 / 0.5 = 10
    r[2, 0] = -5.0 # V(2) = -5 / 0.5 = -10
    
    pi = np.ones((n_states, n_actions))
    
    V, Var = distributional_value_function(P, r, pi, discount)
    
    assert np.isclose(V[1], 10.0)
    assert np.isclose(V[2], -10.0)
    
    # V(0) = 0 + 0.5 * (0.5 * 10 + 0.5 * -10) = 0
    assert np.isclose(V[0], 0.0)
    
    # Variance at 0:
    # Z(0) = 0 + 0.5 * Z(S')
    # Z(S') is either Z(1) (value 10) or Z(2) (value -10) with prob 0.5
    # Var(Z(0)) = 0.5^2 * Var(Z(S'))
    # Var(Z(S')) = E[Z(S')^2] - E[Z(S')]^2
    # E[Z(S')] = 0
    # E[Z(S')^2] = 0.5 * 100 + 0.5 * 100 = 100
    # Var(Z(S')) = 100
    # Var(Z(0)) = 0.25 * 100 = 25
    
    # Wait, Z(1) and Z(2) are deterministic streams of rewards?
    # Z(1) = 5 + 0.5*5 + ... = 10. Variance 0.
    # Z(2) = -10. Variance 0.
    
    assert np.isclose(Var[1], 0.0, atol=1e-4)
    assert np.isclose(Var[2], 0.0, atol=1e-4)
    assert np.isclose(Var[0], 25.0, atol=1e-4)
