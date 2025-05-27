import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import mdp.utils as utils

def hyperbolic_polytope():
    # https://arxiv.org/abs/1902.06865
    n_states, n_actions = 2, 2
    N = 21
    pis = utils.gen_grid_policies(N)
    mdp = utils.build_random_mdp(n_states, n_actions, None)

    n = 10
    discounts = np.linspace(0.1, 1-1e-4, n)
    Vs = []
    for discount in discounts:
        Vs.append((1-discount)*utils.polytope(mdp.P, mdp.r, discount, pis))

    h_V = sum(Vs)/n

    plt.subplot(2, 1, 1)
    plt.scatter(h_V[:, 0], h_V[:, 1])
    plt.subplot(2, 1, 2)
    V = (1-0.9)*utils.polytope(mdp.P, mdp.r, 0.9, pis)
    plt.scatter(V[:, 0], V[:, 1])
    # plt.show() # Replaced with close
    plt.close()


def generate_discounted_polytopes_forvideo():
    """
    ffmpeg -framerate 10 -start_number 0 -i disc%d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
    """
    n_states = 2
    n_actions = 2

    N = 100
    n = 4
    M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,31)]
    Prs = [generate_rnd_problem(n_states,n_actions)for _ in range(n*n)]

    for i, discount in enumerate(np.linspace(0, 1-1e-4,N)):
        print(i)
        plt.figure()
        for j in range(n*n):
            ax = plt.subplot(n,n,j+1)
            P, r = Prs[j]
            Vs = np.hstack([value_functional(P, r, M_pi, discount) for M_pi in M_pis])
            fig = plt.plot(Vs[0, :], Vs[1, :], 'b.')[0]
            ax.set_xlim(np.min(Vs[0, :]),np.max(Vs[0, :]))
            ax.set_ylim(np.min(Vs[1, :]),np.max(Vs[1, :]))
            plt.title('{:.4f}'.format(discount))

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.savefig('figs/discount_experiment_disc{}.png'.format(i))
        plt.close()

def generate_discounted_polytopes():
    n_states = 2
    n_actions = 2

    ny = 6
    nx = 12
    M_pis = [generate_Mpi(n_states, n_actions, pi) for pi in gen_grid_policies(2,2,31)]
    Prs = [generate_rnd_problem(n_states,n_actions)for _ in range(ny)]
    count = 0
    plt.figure(figsize=(16,16))
    for j in range(ny):
        print(j)
        for i, discount in enumerate(np.linspace(0, 1-1e-4,nx)):
            count += 1
            ax = plt.subplot(ny,nx,count)

            P, r = Prs[j]
            Vs = np.hstack([value_functional(P, r, M_pi, discount) for M_pi in M_pis])
            pVs = [density_value_functional(0.1, P, r, M_pi, 0.9) for M_pi in M_pis]

            fig = plt.scatter(Vs[0, :], Vs[1, :], c=pVs)
            ax.set_xlim(np.min(Vs[0, :]),np.max(Vs[0, :]))
            ax.set_ylim(np.min(Vs[1, :]),np.max(Vs[1, :]))

            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    # plt.savefig('../pictures/figures/discounts.png') # This was commented out
    # plt.show() # Replaced with close
    plt.close()

def discount_trajectories():
    """
    Plot the trajectory of the different deterministic policies
    """
    n_states, n_actions = 2, 6

    M_pis = [generate_Mpi(n_states, n_actions, p) for p in get_deterministic_policies(n_states, n_actions)]
    fig = plt.figure(figsize=(16, 16))
    # ax = fig.add_subplot(111)
    n = 20
    P, r = generate_rnd_problem(n_states, n_actions)
    discounts = np.linspace(0.1, 0.999, n)

    Vs = []
    for i in range(n):
        V = np.hstack([value_functional(P, r, M_pi, discounts[i]) for M_pi in M_pis])
        Vs.append(V/np.max(V))
    Vs = np.stack(Vs, axis=-1)
    # print(np.stack(Vs).shape)
    # d x n_M_pi x n
    colors = cm.viridis(np.linspace(0, 1, n))


    for x, y, c in zip(Vs[0, :, :].T, Vs[1, :, :].T, colors):
        plt.scatter(x, y, c=c)
    # plt.show()
    plt.xlabel('V2. max normed')
    plt.ylabel('V1. max normed')
    plt.title('A random {}-state, {}-action MDP'.format(n_states, n_actions))
    plt.savefig('figs/policy_discount_trajectories.png')
    plt.close()

def discount_partitions():
    """
    Plot the partitions of discounts wrt optimal policies.

    Ok, so a MDP that has many changes of the optimal policy wrt to discounts is
    a 'harder' one than one that has few changes.
    Can solve once / few times and transfer the policy between different discounts.
    When / why can this be done?
    """
    n_states, n_actions = 3, 3
    n = 100
    discounts = np.linspace(0.1, 0.999, n)


    for _ in range(10):
        P, r = generate_rnd_problem(n_states, n_actions)
        stars = []
        for i in range(n):
            stars.append(solve(P, r, discounts[i]))

        diffs = [np.sum(np.abs(stars[i]-stars[i+1])) for i in range(n-1)]
        plt.plot(discounts[:-1], np.cumsum(diffs))

    plt.title('Discount partitions')
    plt.xlabel('Discount')
    plt.ylabel('Cumulative changes to the optimal policy')

    # plt.show() # Replaced with close
    plt.savefig('figs/discount_partitions.png')
    plt.close()


if __name__ == '__main__':
    # hyperbolic_polytope()
    discount_partitions()
