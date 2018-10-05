Notes based on https://medium.freecodecamp.org/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc

Policy-based approach:
In this case, we have a policy which will need to optimize. The policy will help us to map from states to actions.
There are two kinds:
- Deterministic: A policy at a given state (s) will always return the same action (a) S = s -> A = a
- Stochastic: It gives a distribution of probability given over different action i.e. Stochastic policy p(A = a|S = s)

Value-based: The goal of the agent is to optimize the value function V(s) which is defined as a function to tells us the maximum
epxected future reward that the agent shall get at each state


The agent will use this value function to select which state to choose at each step. The agent will always take the state with the biggest
value. 
