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
- [x] `trajectory_experiments.py`: What do the trajectories of momentum and over parameterised optimisations look like on a polytope?
  - [ ] How does momentum change the trajectories?
  - [ ] How does over parameterisation yield acceleration? ANd how does its trajectories relate to optimisation via momentum?

### Generalisation



### LMDPs

- [ ] `lmdp_experiments.py`: How do LMDPs compare to MDPs?
  - [ ] Do they give similar results?
  - [ ] How do they scale?
  - [ ] ?

### Other possible experiments

- [ ] `graph_signal_vis.py` Generate a graph of update transitions under an update fn. The nodes will be the value of the deterministic policies. This could be a way to visualise in higher dimensins!? Represent the polytope as a graph. And the value is a signal on the graph. Need a way to take V_pi -> \sum a_i . V_pi_det_i. Connected if two nodes are only a single action different.

***

- [ ] `reproduce_abstraction.py`: Simply reproduce the figures from [Near Optimal Behavior via Approximate State Abstraction](https://arxiv.org/abs/1701.04113)
- [ ] `symmetry_detection.py`: Reproduce results from [Theja](???)
