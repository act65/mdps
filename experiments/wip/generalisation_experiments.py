

def generate_avi_vs_vi(mdp):
    print('\nRunning AVI vs VI')
    lr = 0.1

    # vi
    init = rnd.standard_normal((mdp.S, mdp.A))
    qs = utils.solve(value_iteration(mdp, lr), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    n = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='g', label='vi')
    plt.scatter(vs[:, 0], vs[:, 1], c=range(n), cmap='spring', s=10)

    # avi
    # K = np.eye(n_states)
    K = rnd.standard_normal((mdp.S))
    K = (K + K.T)/2 + 1  # this can accelerate learning. but also lead to divergence. want to explore this more!!!
    d = rnd.random(mdp.S)
    D = np.diag(d/d.sum())  # this can change the optima!!
    # D = np.eye(n_states)
    qs = utils.solve(adjusted_value_iteration(mdp, lr, D, K), init)
    vs = np.vstack([np.max(q, axis=1) for q in qs])

    m = vs.shape[0]
    plt.scatter(vs[0, 0], vs[0, 1], c='r', label='avi')
    plt.scatter(vs[:, 0], vs[:, 1], c=range(m), cmap='autumn', s=10)
    plt.title('VI: {}, Avi {}'.format(n, m))
    plt.legend()
    plt.colorbar()

    plt.savefig('figs/avi-vs-vi.png')
    plt.close()
