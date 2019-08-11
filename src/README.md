
Supposedly the iterations required to solve a LMDP are independent of |S|, |A|.
What is the dependence on $\gamma$?
Could do experiments?

What about the trajectory that the LMDP solver takes through the polytope??


***

Goals.

- Understand MDPs and their properties
- Which properties are important?
- How can we put metrics on these properties? And analyse in higher dims?
- ?

Todos.

- Generalise the types of parameterised fns (jax must have some tools for this)
- calculate the tangent fields. are they similar? what about for parameterised versions, how can we calculate their vector fields???
- use a random seed for reproducability.

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
