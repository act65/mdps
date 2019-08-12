This is an attmept to get some better intution about [MDPs](https://en.wikipedia.org/wiki/Markov_decision_process). Their properties, geometry, dynamics...

This work is a mixture of;

- [The Value Function Polytope in Reinforcement Learning](https://arxiv.org/abs/1901.11524)
- [Towards Characterizing Divergence in Deep Q-Learning](https://arxiv.org/abs/1903.08894)
- [Implicit Acceleration by Overparameterization](https://arxiv.org/abs/1802.06509)
- [Efficient computation of optimal actions](https://www.pnas.org/content/106/28/11478)

***

Setting;
- All experiments are done with tabular MDPs.
- The rewards are deterministic and the transtions are stochastic.
- All setting assume we can make synchronous updates.
- It is assumed we have access to the transition function and reward function.
- ?

***

If you want to __reproduce__ my results.
TODO: use a random seed.

First, install...

```python
python setup.py install
```

Then, you should be able to run each of the scripts in `experiments/` and generate all the figures in `figs/`.

## Experiments

### Density

- [ ] `density_experiments.py`: How are policies distributed in value space?
  - [x] Visualise the density of the value function polytope.
  - [ ] Calculate the expected suboptimality (for all policies - and all possible Ps/rs)? How does this change in high dimensions?
  - [ ] Apply an abstraction (with X property) and visualise how the distribution of value changes.
  - [ ] Visualise a distributional value polytope.

### Discounting

- [ ] `discounting_experiments.py`:
  - [ ] Visulise how changing the discount rate changes the shape of the polytope.
  - [ ] Explore and visualise hyperbolic discounting.
  - [ ] How does the discount change the optimal policy?

### Search dynamics

- [ ] `partition_experiments.py`: How do different optimisers partition the value / policy space?
  - [x] Visualise how the number of steps required for GPI partitions the policy / value spaces.
  - [ ] Generalise GPI to work in higher dimensons. Calclate how does it scales.
  - [ ] Visualise n steps for PG / VI.
  - [ ] calculate the tangent fields. are they similar? what about for parameterised versions, how can we calculate their vector fields???
- [x] `trajectory_experiments.py`: What do the trajectories of momentum and over parameterised optimisations look like on a polytope?
  - [ ] How does momentum change the trajectories?
  - [ ] How does over parameterisation yield acceleration? ANd how does its trajectories relate to optimisation via momentum?
  - [ ] Test dynamics with complex valued parameters.
  - [ ] Generalise the types of parameterised fns (jax must have some tools for this)

### Generalisation

- [ ] `generalisation.py`: How does generalsation accelerate the dynamics / learning?
  - [ ] Use the NTK to explore trajectories when generalisation does / doesnt make sense.
  - [ ] ???

### LMDPs

- [ ] `lmdp_experiments.py`: How do LMDPs compare to MDPs?
  - [ ] Do they give similar results?
  - [ ] How do they scale?
  - [ ] ?

### Other possible experiments

- [ ] `graph_signal_vis.py` Generate a graph of update transitions under an update fn. The nodes will be the value of the deterministic policies. This could be a way to visualise in higher dimensins!? Represent the polytope as a graph. And the value is a signal on the graph. Need a way to take V_pi -> \sum a_i . V_pi_det_i. Connected if two nodes are only a single action different.


### Other facets of MDPs not explored

- The effects of exploration, sampling and buffers.
  - How the effect stability of the dynamics
  - How the bias the learning
  - ?
- Model-based solvers (MPC style)
- ?


***


Questions
- Is param + momentum dynamics are more unstable? Or that you move around value-space in non-linear ways??
- Is param + momentum only faster bc it is allowed larger changes? (normalise for the number of updates being made). __Answer:__ No. PPG is still faster.
- What if we make the learning rate very small? (!!?!) (plot momentum as a fn of different lrs. w same init / mdp)
- What is the max difference between a trajectory derived from cts flow versus a trajectory of discretised steps on the same gradient field?
- What happens if we take two reparameterisations of the same matrix? Are their dynamics different? __Answer:__ No.
- What are the ideal dynamics? PI jumps around. VI travels in straight lines, kinda.

Observations

- Parameterisation seems to accelerate PG, but not VI. Why?
- Sometimes VI converges to a weird place.
- With smaller and smaller learning rates, and fixed decay, the dynamics of momentum approach GD.
