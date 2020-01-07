"""
1. Make a mdp solver that uses thompson sampling
2. augment the thompson sampler with an inductive bias

How to generalise this for deep RL?
- our measure of complexity works on tabular representations of the transition / reward fns
-

This doesnt make sense. Not exploring...

"""
import mpd.utils as utils
import jax.numpy as np
from jax import grad, jit
import numpy.random as rnd

def reparam_sample(mu, var):
    e = rnd.standard_normal(var.shape)
    return mu + e * var

def init_params(nS, nA):
    nP = nS*nS*nA
    nr = nS*nA
    return rnd.standard_normal((nP*2 + nr*2,))

def sample_model(params):
    P_mean, P_stdev, r_mean, r_stdev = parse_params(params, nS, nA)
    P_unnormed = reparam_sample(P_mean, P_stdev**2)
    r = reparam_sample(r_mean, r_stdev**2)
    return P_unnormed/np.sum(P_unnormed, axis=0), r

def parse_params(params, nS, nA):
    nP = nS*nS*nA
    nr = nS*nA
    P_mean = params[:nP].reshape((nS,nS,nA))
    P_stdev = params[nP:nP*2].reshape((nS,nS,nA))  # what about covar?
    r_mean = params[nP*2:nP*2 + nr].reshape((nS,nA))
    r_stdev = params[nP*2 + nr:].reshape((nS,nA))
    return P_mean, P_stdev, r_mean, r_stdev

def gaussian_density_fn(mean, stdev, x):
    return (1/(stdev*np.sqrt(2*np.pi))) * np.exp(-0.5*np.square((mean-x)/stdev))

def ML_loss(params, obs):
    P_mean, P_stdev, r_mean, r_stdev = parse_params(params)
    obs_P, obs_r = obs

    p_obs_P = gaussian_density_fn(P_mean, P_stdev, obs_P)
    p_obs_r = gaussian_density_fn(r_mean, r_stdev, obs_r)

    return -np.sum(np.log(p_obs_P) + np.log(p_obs_r))


def TS(mdp, init, lr=0.01):
    dLdp = grad(ML_loss)
    counter = 0

    def update_fn(params):
        obs = sample_model()
        params -= lr*dLdp(params, obs)
        counter += 1

    return update_fn

if __name__ == '__main__':

    n_states = 12
    n_actions = 2

    mdp = utils.build_random_mdp(n_states, n_actions, 0.5)

    params = init_params(n_states, n_actions)
