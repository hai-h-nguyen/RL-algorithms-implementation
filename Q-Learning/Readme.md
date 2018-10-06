Introducing Q-table:

It is a simple look-up table where we calculate the maximum expected future rewards for action at each state. This table will guide us
to the best action at each state.

I implement the Q-table and tested with 3 environments in Gym: Taxi-v2, Frozen-v0 and Frozen8x8-v0. One note is that the exploration/exploitation
tradeoff should be dependent on the number of training episodes. I reduce episilon linearly with the number of training episodes. Initially, epsilon
is max_epsilon (1.0) but it will become min_epsilon towards the end of the training.