This is an attmept to get some better intution about [MDPs](https://en.wikipedia.org/wiki/Markov_decision_process). Insight into their properties; structure, geometry, dynamics, ...

This work is inspired by;

- [The Value Function Polytope in Reinforcement Learning](https://arxiv.org/abs/1901.11524)
- [Towards Characterizing Divergence in Deep Q-Learning](https://arxiv.org/abs/1903.08894)
- [Implicit Acceleration by Overparameterization](https://arxiv.org/abs/1802.06509)
- [Efficient computation of optimal actions](https://www.pnas.org/content/106/28/11478)

## Setting

- All experiments are done with tabular MDPs.
- The rewards and transtions are stochastic.
- We use synchronous observations and updating.
- It is sometimes assumed we have access to the transition function and reward function.

## Install

```
pip install git+https://github.com/act65/mdps.git
```

Or. If you want to develop some new experiments.

```
git clone
cd mdps
pip install -r requirements.txt
python setup.py develop
```

## Reproduce

I used a random seed in my experiments so you should be able to run each of the scripts in `experiments/` and reproduce all the figures in `figs/`.

## Experiments

### Density

- [ ] `density_experiments.py`: How are policies distributed in value space?
  - [x] Visualise the density of the value function polytope.
  - [ ] Calculate the expected suboptimality (for all policies - and all possible Ps/rs)? How does this change in high dimensions?

### Discounting

- [ ] `discounting_experiments.py`:
  - [x] Visualise how changing the discount rate changes the shape of the polytope.
  - [ ] How does the discount change the optimal policy?
  - [ ] Explore and visualise hyperbolic discounting.

### Search dynamics

- [ ] `iteration_complexity_experiments.py`: How do different optimisers partition the value / policy space?
  - [x] Visualise how the number of steps required for GPI partitions the policy / value spaces.
  - [x] Visualise color map of iteration complexity for for PG / VI and variants.
  - [ ] calculate the tangent fields. are they similar? what about for parameterised versions, how can we calculate their vector fields???
- [x] `trajectory_experiments.py`: What do the trajectories of momentum and over parameterised optimisations look like on a polytope?
  - [x] Visualise how momentum changes the trajectories of different solvers
  - [ ] How does over parameterisation yield acceleration? And how does its trajectories relate to optimisation via momentum?
  - [ ] Test dynamics with complex valued parameters.
  - [ ] Generalise the types of parameterised fns (jax must have some tools for this)
- Other
  - [ ] Generalise GPI to work in higher dimensons. Calclate how does it scales.


### Generalisation

- [ ] `generalisation.py`: How does generalsation accelerate the dynamics / learning?
  - [ ] Use the NTK to explore trajectories when generalisation does / doesnt make sense.
  - [ ] ???

### LMDPs

- [x] `lmdp_experiments.py`: How do LMDPs compare to MDPs?
  - [x] Do they give similar results?
  - [x] What does the linearised TD operator look like (its vector field)?
  - [ ] How do they scale?

### Other possible experiments

- [ ] `graph_signal_vis.py` Generate a graph of update transitions under an update fn. The nodes will be the value of the deterministic policies. This could be a way to visualise in higher dimensins!? Represent the polytope as a graph. And the value is a signal on the graph. Need a way to take V_pi -> \sum a_i . V_pi_det_i. Connected if two nodes are only a single action different.
- [ ] `mc-grap.py`: could explore different ways of estimating the gradient; score, pathwise, measure.
- [ ] Visualise a distributional value polytope.

### Other facets of MDPs not explored

- The effects of exploration, sampling and buffers.
  - How the effect stability of the dynamics
  - How the bias the learning
  - ?

***

Observations

- Parameterisation seems to accelerate PG, but not VI. Why?
- With smaller and smaller learning rates, and fixed decay, the dynamics of momentum approach GD.


Questions

- Is param + momentum dynamics are more unstable? Or that you move around value-space in non-linear ways??
- Is param + momentum only faster bc it is allowed larger changes? (normalise for the number of updates being made). __Answer:__ No. PPG is still faster.
- What is the max difference between a trajectory derived from cts flow versus a trajectory of discretised steps on the same gradient field?
- What happens if we take two reparameterisations of the same matrix? Are their dynamics different? __Answer:__ No. (???)
- What are the ideal dynamics? PI jumps around. VI travels in straight lines, kinda.
