import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from polytope_tools import *

"""
The question that has been bugging me.
If we use the deterministic policies as a basis vs interior policies.

Intuition. The interior policies are (spares) mixtures of the deterministic policies.
Sampling k interior points should allow us to infer m deterministic policies.

No, the deterministic policies need to be sparse!?!?

"""

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)

def converged(xs, n=10):
    if len(xs) <= n:
        return False
    elif len(xs) > 10000:
        raise ValueError('not converged')
    elif np.isnan(xs[-1]).any():
        raise ValueError('Nan')
    else:
        diffs = [np.linalg.norm(xs[-i] - xs[-j]) for i in range(n) for j in range(n) if i > j]
        if sum(diffs)/n > 1e-6:
            return True
        elif sum(diffs)/n > 1e2:
            print('s')
            raise ValueError('diverging')

def l1(x):
    return tf.reduce_sum(tf.abs(x))

def l2(x):
    return tf.reduce_sum(x**2)

def L(x, A, b, alpha=1.0):
    return l2(tf.matmul(A, x) - b) + alpha * l1(x)

def dLdx(x, A, b):
    x = tf.constant(x)
    A = tf.constant(A)
    b = tf.constant(b)

    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = L(x, A, b)

    return tape.gradient(loss, x).numpy()

def solve(A, b, alpha=1.0, lr=0.001):
    """
    Solve an overconstrained, regularised linear system.
    Ax = b.
    Where
    - x should be sparse.
    - shape of A is [n, m] where n << m. and x is shape [m, 1]. and b is shape [n, 1]
    """
    n, m = A.shape
    assert b.shape == (n, 1)

    xs = [np.random.standard_normal((m, 1))]
    while not converged(xs):
    # for _ in range(1000):
        x_t = xs[-1]
        x_tp1 = x_t - lr * dLdx(x_t, A, b)
        xs.append(x_tp1)

    return xs[-1]

def solvev2(A, b):
    x, r, rank, s = np.linalg.lstsq(A, b)
    return x

def spectrogram(P, r, pi, n=10):
    """
    A MDP spectrogram.
    the basis is [discount rate x deterministic policies].
    """
    n_states = P.shape[1]
    n_actions = P.shape[0] // n_states

    discounts = np.linspace(0.5, 0.75, n)

    detMpis = [generate_Mpi(n_states, n_actions, p) for p in get_deterministic_policies(n_states, n_actions)]

    detVs = [np.hstack([value_functional(P, r, M_pi, discount) for M_pi in detMpis])
            for discount in discounts]
    Vs = np.hstack([value_functional(P, r, pi, discount) for discount in discounts])

    energies = np.hstack([solve(detVs[i], Vs[:, i:i+1]) for i in range(n)])
    # energies = list(sorted(list(energies), key=np.linalg.norm))

    return energies.T

def compressed_spectrogram():
    """
    have measurements, y, want to find the x that likely gave them
    max_x || f(x) - y ||

    not sure how that is related to this!?
    max_x || f(x) - y ||
    where y_i is sampled for deterministic policies
    where y_i is sampled for any policy

    but we are trying to infer f?!
    there is a dual way to think about this!?
    """
    n_states, n_actions = 4, 2

    fig = plt.figure(figsize=(16, 16))

    P, r = generate_rnd_problem(n_states, n_actions)

    energy = spectrogram(P, r, generate_rnd_policy(n_states, n_actions))

    plt.imshow(energy)
    plt.ylabel('Discounts')
    plt.xlabel('Det policies')
    plt.show()


def basis_energies():
    """
    y = Ax
    A is the mixture.
    y are the measurements
    x is the value of the deterministic policies
    || Ax - y ||
    """


    n_states, n_actions = 4, 2

    fig = plt.figure(figsize=(16, 16))

    P, r = generate_rnd_problem(n_states, n_actions)
    energies = [spectrogram(P, r, generate_rnd_policy(n_states, n_actions)) for _ in range(10)]

    n, m = energy.shape
    plt.figure()
    for energy in energies:
        for i in range(n):
            plt.subplot(5, 2, i+1)
            plt.scatter(range(m), energy[i]**2)
    plt.show()


def total():
    n_states, n_actions = 6, 2


    P, r = generate_rnd_problem(n_states, n_actions)
    energies = np.stack([spectrogram(P, r, generate_rnd_policy(n_states, n_actions)) for _ in range(20)], axis=-1)

    n = 10
    discounts = np.linspace(0.5, 0.75, n)
    pi_stars = [solve_optimal(P, r, discount) for discount in discounts]
    detMpis = [generate_Mpi(n_states, n_actions, p) for p in get_deterministic_policies(n_states, n_actions)]
    is_star = [np.argmax([np.isclose(star, dpi).all().astype(np.float32) for dpi in detMpis]) for star in pi_stars]
    print(is_star)

    # discounts x det policies x rnd policies
    fig = plt.figure(figsize=(16, 16))
    _,m,_ = energies.shape
    print(energies.shape)
    for i in range(n):
        plt.subplot(5, 2, i+1)
        plt.scatter(range(m), np.mean(energies[i], axis=-1))
        plt.scatter(is_star[i], np.linalg.norm(value_functional(P, r, pi_stars[i], discounts[i])))

    plt.show()


if __name__ == '__main__':
    # compressed_spectrogram()
    total()
