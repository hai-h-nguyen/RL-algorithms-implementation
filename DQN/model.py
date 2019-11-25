import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from replay import ReplayMemory
from collections import deque
from utils import *
from baselines import logger


class DQN(nn.Module):
    def __init__(self, state_dim=5, action_dim=4, *, num_layers=3, hidden_dim=64):
        """Deep Q-Network PyTorch model.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.hidden = nn.ModuleList()
        for _ in range(self.num_layers - 2):
            self.hidden.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.out = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, states) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param states: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of state-space.
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space.  This represents the Q values Q(s, .).
        """
        x = F.relu(self.fc1(states))

        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.out(x)

        return x

    # utility methods for cloning and storing models.  DO NOT EDIT
    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.state_dim, self.action_dim),
            'kwargs': {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
            },
            'state_dict': self.state_dict(),
        }


def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
    """Perform a single batch-update step on the given DQN model.
    :param optimizer: nn.optim.Optimizer instance.
    :param batch:  Batch of experiences (class defined earlier).
    :param dqn_model:  The DQN model to be trained.
    :param dqn_target:  The target DQN model, ~NOT~ to be trained.
    :param gamma:  The discount factor.
    :rtype: float  The scalar loss associated with this batch.
    """
    state_batch = batch.states
    action_batch = batch.actions
    reward_batch = batch.rewards
    next_state_batch = batch.next_states
    done_batch = batch.dones
    done_batch = done_batch.float()

    values = dqn_model(state_batch).gather(1, action_batch)
    target_values = reward_batch + gamma * dqn_target(next_state_batch).max(1)[0].detach().view(-1, 1) * (1.0 - done_batch)

    assert (
        values.shape == target_values.shape
    ), 'Shapes of values tensor and target_values tensor do not match.'

    # testing that the value tensor requires a gradient,
    # and the target_values tensor does not
    assert values.requires_grad, 'values tensor should require gradients'
    assert (
        not target_values.requires_grad
    ), 'target_values tensor should require gradients'

    # computing the scalar MSE loss between computed values and the TD-target
    loss = F.mse_loss(values, target_values)

    optimizer.zero_grad()  # reset all previous gradients
    loss.backward()  # compute new gradients
    optimizer.step()  # perform one gradient descent step

    return loss.item()


def train_dqn(
    env,
    num_steps,
    *,
    replay_size,
    batch_size,
    exploration,
    gamma, 
    train_freq=1,
    print_freq=100, 
    target_network_update_freq=500,
    t_learning_start=1000):
    """
    DQN algorithm.

    Compared to previous training procedures, we will train for a given number
    of time-steps rather than a given number of episodes.  The number of
    time-steps will be in the range of millions, which still results in many
    episodes being executed.

    Args:
        - env: The openai Gym environment
        - num_steps: Total number of steps to be used for training
        - replay_size: Maximum size of the ReplayMemory
        - batch_size: Number of experiences in a batch
        - exploration: a ExponentialSchedule
        - gamma: The discount factor

    Returns: (saved_models, returns)
        - saved_models: Dictionary whose values are trained DQN models
        - returns: Numpy array containing the return of each training episode
        - lengths: Numpy array containing the length of each training episode
        - losses: Numpy array containing the loss of each training batch
    """
    # check that environment states are compatible with our DQN representation
    assert (
        isinstance(env.observation_space, gym.spaces.Box)
        and len(env.observation_space.shape) == 1
    )

    # get the state_size from the environment
    state_size = env.observation_space.shape[0]

    # initialize the DQN and DQN-target models
    dqn_model = DQN(state_size, env.action_space.n)
    dqn_target = DQN.custom_load(dqn_model.custom_dump())

    # initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters(), lr=5e-4)

    # initialize the replay memory
    memory = ReplayMemory(replay_size, state_size)

    # initiate lists to store returns, lengths and losses
    rewards = []
    returns = []
    lengths = []
    losses = []

    last_100_returns = deque(maxlen=100)
    last_100_lengths = deque(maxlen=100)

    # initiate structures to store the models at different stages of training
    saved_models = {}

    i_episode = 0  
    t_episode = 0

    state = env.reset()

    # iterate for a total of `num_steps` steps
    for t_total in range(num_steps):
        # use t_total to indicate the time-step from the beginning of training
        
        if t_total >= t_learning_start:
            eps = exploration.value(t_total - t_learning_start)
        else:
            eps = 1.0
        action = select_action_epsilon_greedy(dqn_model, state, eps, env)
        next_state, reward, done, _ = env.step(action)
        memory.add(state, action, reward, next_state, done)

        rewards.append(reward)
        state = next_state
        
        if t_total >= t_learning_start and t_total % train_freq == 0:
            batch = memory.sample(batch_size)
            loss = train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma)
            losses.append(loss)

        # update target network
        if t_total >= t_learning_start and t_total % target_network_update_freq == 0:
            dqn_target.load_state_dict(dqn_model.state_dict())

        if done:

            # Calculate episode returns
            G = 0
            for i in range(len(rewards)):
                G += rewards[i] * pow(gamma, i)
           
            # Collect results
            lengths.append(t_episode + 1)
            returns.append(G)

            last_100_returns.append(G)
            last_100_lengths.append(t_episode + 1)

            if i_episode % print_freq == 0:
                logger.record_tabular("time step", t_total)

                logger.record_tabular("episodes", i_episode)
                logger.record_tabular("step", t_episode + 1)
                logger.record_tabular("return", G)
                logger.record_tabular("mean reward", np.mean(last_100_returns))
                logger.record_tabular("mean length", np.mean(last_100_lengths))

                logger.record_tabular("% time spent exploring", int(100 * eps))
                logger.dump_tabular()

            # End of episode so reset time, reset rewards list
            t_episode = 0
            rewards = []

            # Environment terminated so reset it
            state = env.reset()

            # Increment the episode index
            i_episode += 1

        else:
            t_episode += 1

    return (
        dqn_model,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
    )