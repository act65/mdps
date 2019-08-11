import numpy as np
import numpy.random as rnd
import itertools

def generate_rnd_problem(n_states, n_actions):
    P = rnd.random((n_states * n_actions, n_states))**2
    P = P/P.sum(axis=1, keepdims=True)
    r = rnd.random((n_states * n_actions, 1))
    return P, r

def value_functional(P, r, M_pi, discount):
    """
    V = r_{\pi} + \gamma P_{\pi} V
      = (I-\gamma P_{\pi})^{-1}r_{\pi}

    Args:
        P ():
        r ():
        M_pi ():
        discount (float): the temporal discount value
    """
    n = P.shape[-1]
    P_pi = np.dot(M_pi, P)
    r_pi = np.dot(M_pi, r)
    return np.dot(np.linalg.inv(np.eye(n) - discount*P_pi), r_pi)

def density_value_functional(px, P, r, M_pi, discount):
    P_pi = np.dot(M_pi, P)
    r_pi = np.dot(M_pi, r)

    J = value_jacobian(r_pi, P_pi, discount)
    return probability_chain_rule(px, J)

def value_jacobian(r_pi, P_pi, discount):
    """
    Returns:
        [inputs x outputs] ???
    """
    return r_pi * (np.eye(P_pi.shape[0]) - discount * P_pi)**(-2)

def entropy_jacobian(pi):
    """
    H(pi) = - sum p log p
    dHdpi(j) = 1 + log p
    """
    return -1 - np.log(pi)

def probability_chain_rule(px, J):
    """
    p(f(x)) = abs(|J|)^-1 . p(x)
    """
    return (np.abs(np.linalg.det(J))**(-1)) * px

def get_pi(M_pi):
    n, m = M_pi.shape
    n_actions = m // n
    pi = np.zeros((n, n_actions))
    for i in range(n):
        for j in range(n_actions):
            pi[i, j] = M_pi[i, i*n_actions + j]
    return pi


#########################################################


def generate_Mpi(n_states, n_actions, ps):
    """
    A policy is represented by a block diagonal |S| x |S||A| matrix M.

    Args:
        n_states (int): the number of states
        n-actions (int): the number of actions
        ps (array[n_states, n_actions]): the probabilities of taking action a in state s.

    Returns:
        (array[n_states, n_states x n_actions])
    """
    A = np.ones((1, n_actions))
    S = np.eye(n_states)

    M_pi = np.zeros((n_states, n_states * n_actions))
    M_pi[np.where(np.equal(1, np.kron(S, A)))] = ps.reshape(-1)
    return M_pi

def pi(M_pi, s, a):
    """
    Let state s be indexed by i and the action a be indexed by j.
    Then we have that M[i, i x |A| + j] = pi(a|s)
    """
    return M_pi[s, s*n_actions+a]

def normalise(x, axis):
    return x/np.sum(x, axis=axis, keepdims=True)

def exp_dist(x, lambda_=3.5):  # why 3.5!?
    return lambda_*np.exp(-lambda_*x)

def uniform_simplex(shape):
    # god damn. it's not as simple as I thought to generate uniform distributions on simplexs
    # https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
    # http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
    return normalise(exp_dist(rnd.random(shape)),axis=1)

def generate_rnd_policy(n_states, n_actions):
    return generate_Mpi(n_states, n_actions, uniform_simplex((n_states, n_actions)))

# def gen_grid_policies(n_states, n_actions, N=11):
#     # TODO need to generalise to nD
#     # only works for 2 states, 2 actions
#     x = np.linspace(0, 1, N)
#     return [np.array([x[i],1-x[i],x[j],1-x[j]]) # will not generalise to n states
#                   for i in range(N)
#                   for j in range(N)]

def gen_grid_policies(n_states, n_actions, N=11):
    """
    This scales badly!!!
    """
    simplicies = list(itertools.product(np.linspace(0, 1, N), repeat=n_actions))
    simplicies = [s/np.sum(s) for s in simplicies]
    return [np.vstack(x) for x in itertools.product(*[simplicies for _ in range(n_states)])]

def get_deterministic_policies(n_states, n_actions):
    simplicies = list([np.eye(n_actions)[i] for i in range(n_actions)])
    pis = list(itertools.product(*[simplicies for _ in range(n_states)]))
    return [np.stack(p) for p in pis]

def most_common(lst):
    data = collections.Counter(lst)
    return max(lst, key=data.get)

def check_differences(policies):
    m = len(policies)
    count = 0
    for i in range(m):
        for j in range(m):
            if i > j:
                if not np.equal(policies[i], policies[j]).all():
                    count += 1
    return count

def solve_optimal(P, r, discount, n=5):
    # n the number of retries
    # generate optimal pi
    n_states = P.shape[-1]
    n_actions = P.shape[0]//n_states
    stars = []

    done = False
    while not done:
        Mpi = generate_rnd_policy(n_states, n_actions)
        pis, vs = solve(policy_iteration_update, P, r, Mpi, discount, atol=1e-5, maxiters=20)
        pi_star = pis[-1]
        stars.append(pi_star)
        diffs = check_differences(stars[-n:])
        if diffs == 0 and len(stars) > n:
            done = True
    return stars[-1]


#######################################

def softmax(x):
    assert len(x.shape) == 2
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def onehot(x, n):
    return np.eye(n)[x]

def solve(update_fn, P, r, Mpi, discount, atol=1e-4, maxiters=10):
    """
    Generates the dynamics of update_fn .
    Initial condition = Mpi

    An ODE solver!?
    """
    converged = False
    pis = [Mpi]
    vs = [value_functional(P, r, Mpi, 0.9)]
    count = 0
    while not converged:
        Mpi = update_fn(P, r, Mpi, discount)
        V = value_functional(P, r, Mpi, 0.9)

        if np.isclose(vs[-1] - V, np.zeros(V.shape)).all():
            converged = True
        else:
            vs.append(V)
            pis.append(Mpi)


        count += 1
        if count > maxiters-2:
            break

    return pis, vs

def greedy_solution(V, P):
    # TODO this isnt correct?! should be: argmax_a r + gamma P V?!
    n_states = P.shape[-1]
    n_actions = P.shape[0]//P.shape[-1]
    EV = np.dot(P, V).reshape((n_states, n_actions))  # expected value of each action in each state

    return generate_Mpi(n_states, n_actions, onehot(np.argmax(EV,axis=1), n_actions))

def policy_iteration_update(P, r, M_pi, gamma):
    V = value_functional(P, r, M_pi, gamma)
    return greedy_solution(V, P)

def interior_iteration_update(P, r, M_pi, gamma):
    """
    How can we get more information about the shape of the polytope by
    tranversing the interior, rather than jumping to the edges.

    Like a compressed sensing problem?! Take a set of random actions,
    and mix together the directions.
    """
    pass
