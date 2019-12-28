import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from replay_recurrent import RecurrentReplayMemory
from collections import deque
from utils import *
from baselines import logger
import random
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DRQN(nn.Module):
    def __init__(self, obs_dim=2, action_dim=4, hidden_dim=512):
        """Deep Recurrent Q-Network PyTorch model.

        Args:
            - obs_dim: Dimensionality of observations
            - action_dim: Dimensionality of actions
        """

        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.lstm = nn.LSTM(input_size = self.hidden_dim, hidden_size = 512, num_layers = 1, batch_first = True)
        self.fc4 = nn.Linear(512, self.action_dim)

    def forward(self, observations, bsize, episode_length, hidden_state, cell_state) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param obs: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of observation-space.
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space.  This represents the Q values Q(s, .).
        """
        observations = observations.view(bsize * episode_length, 1, self.obs_dim)
        x = F.relu(self.fc1(observations))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = x.view(bsize, episode_length, 512)
        lstm_out = self.lstm(x, (hidden_state, cell_state))
        out = lstm_out[0][:, episode_length-1, :]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]

        x = self.fc4(out)

        return x, (h_n, c_n)     
        
    def init_hidden_states(self, bsize):
        """Init hidden state values.

        :param bsize: batch_size
        :rtype: zeros tensors
        """

        h = torch.zeros(1, bsize, 512).float().to(device)
        c = torch.zeros(1, bsize, 512).float().to(device)
        
        return h,c

def train_drqn_batch(optimizer, batch, episode_training_len, dqn_model, dqn_target, gamma):
    """Perform a single batch-update step on the given DQN model.
    :param optimizer: nn.optim.Optimizer instance.
    :param batch:  Batch of experiences (class defined earlier).
    :param dqn_model:  The DQN model to be trained.
    :param dqn_target:  The target DQN model, ~NOT~ to be trained.
    :param gamma:  The discount factor.
    :rtype: float  The scalar loss associated with this batch.
    """    
    current_states = []
    acts = []
    rewards = []
    next_states = []
    dones = []

    batch_size = len(batch)

    hidden_batch, cell_batch = dqn_model.init_hidden_states(batch_size)

    for b in batch:
        cs, ac, rw, ns, ds = [],[],[],[],[]
        for element in b:
            cs.append(element[0])
            ac.append(element[1])
            rw.append(element[2])
            ns.append(element[3])
            ds.append(element[4])
        current_states.append(cs)
        acts.append(ac)
        rewards.append(rw)
        next_states.append(ns)
        dones.append(ds)
    
    current_states = np.array(current_states)
    acts = np.array(acts)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
                                
    torch_current_states = torch.from_numpy(current_states).float().to(device)
    torch_acts = torch.from_numpy(acts).long().to(device)
    torch_rewards = torch.from_numpy(rewards).float().to(device)
    torch_next_states = torch.from_numpy(next_states).float().to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    Q_next, _ = dqn_target.forward(torch_next_states, batch_size, episode_training_len, hidden_batch, cell_batch)
    Q_next_max, _ = Q_next.detach().max(dim = 1)

    target_values = torch_rewards[:, episode_training_len - 1] + (gamma * Q_next_max) * (1 - dones[:, -1])
    
    Q_s, _ = dqn_model.forward(torch_current_states, batch_size, episode_training_len, hidden_batch, cell_batch)
    Q_s_a = Q_s.gather(dim=1,index=torch_acts[:, episode_training_len - 1].unsqueeze(dim = 1)).squeeze(dim = 1)

    # testing that they share the same shapes
    assert (
        Q_s_a.shape == target_values.shape
    ), 'Shapes of values tensor and target_values tensor do not match.'    
    
    # testing that the value tensor requires a gradient,
    # and the target_values tensor does not
    assert Q_s_a.requires_grad, 'values tensor should require gradients'
    assert (
        not target_values.requires_grad
    ), 'target_values tensor should require gradients'

    loss = F.mse_loss(Q_s_a, target_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def test_dqn(
    model_path,
    env, 
    num_test_episodes, 
    gamma,
    render, 
    episode_max_len=30
    ):
    """
    DQN testing algorithm.

    Args:
        - model_path: checkpoint file
        - env: The openai Gym environment
        - num_test_episodes: Total number of steps to be used for testing
        - gamma: The discount factor
        - episode_max_len: Go beyonds this and the episode is deemed not successful
        - render: render or not

    Returns: None
    """    

    # initialize the DQN model
    dqn_model = DRQN().float().to(device)

    # load weights from file
    weights = torch.load(model_path)
    dqn_model.load_state_dict(weights)

    returns = []
    lens = []

    episode_cnt = 0

    # start testing for a number of episodes
    while episode_cnt < num_test_episodes:

        prev_state = env.reset()
        # Extract the angle and cart position
        prev_state = prev_state[[0, 2]]

        episode_return = 0
        rewards = []
        episode_len_count = 0

        hidden_state, cell_state = dqn_model.init_hidden_states(bsize=1)

        while episode_len_count < episode_max_len:
            
            episode_len_count += 1
            
            torch_x = torch.from_numpy(prev_state).float().to(device)
            model_out = dqn_model.forward(torch_x, 1, 1, hidden_state, cell_state)
            out = model_out[0]
            action = int(torch.argmax(out[0]))
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
            
            next_state, reward, done, _ = env.step(action)
            if render == 1:
                env.render()
            next_state = next_state[[0, 2]]
            
            rewards.append(reward)
            
            prev_state = next_state            
                
            if done:
                episode_cnt += 1

                for i in range(len(rewards)):
                    episode_return += rewards[i] * pow(gamma, i)

                lens.append(len(rewards))
                returns.append(episode_return)
                
                break

    print("Test {} episodes:".format(num_test_episodes))
    print("Mean len: {:.2f}, Mean returns: {:.2f}".format(np.mean(lens), np.mean(returns)))

def train_dqn(
    env,
    num_timesteps,
    *,
    replay_size,
    batch_size,
    exploration,
    gamma, 
    train_freq=1,
    print_freq=100,
    model_save_freq=500,
    target_network_update_freq=500,
    num_prepopulate_episode=100, 
    episode_training_len=8, 
    episode_max_len=30):
    """
    DQN algorithm.

    Compared to previous training procedures, we will train for a given number
    of time-steps rather than a given number of episodes.  The number of
    time-steps will be in the range of millions, which still results in many
    episodes being executed.

    Args:
        - env: The openai Gym environment
        - num_episodes: Total number of steps to be used for training
        - replay_size: Maximum size of the ReplayMemory
        - batch_size: Number of experiences in a batch
        - exploration: a ExponentialSchedule
        - gamma: The discount factor
        - train_freq:
        - print_freq:
        - target_network_update_freq:
        - num_prepopulate_episode:
        - t_time_steps
        - t_max_steps

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

    # initialize the DQN and DQN-target models
    dqn_model = DRQN().float().to(device)
    dqn_target = DRQN().float().to(device)

    dqn_target.load_state_dict(dqn_model.state_dict())

    # initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters(), lr=5e-4)

    # initialize the replay memory and prepopulating with some episodes
    replay_buffer = RecurrentReplayMemory(replay_size, episode_training_len, episode_max_len)
    replay_buffer.pre_populate(env, num_prepopulate_episode)

    last_100ep_returns = deque(maxlen=100)
    last_100ep_lens = deque(maxlen=100)

    losses = []

    t_total_steps = 0
    t_episode = 0
    short_episode_cnt = 0
    good_episode_cnt = 0

    # start training for a number of time steps
    while t_total_steps < num_timesteps:

        prev_state = env.reset()
        # Extract the angle and cart position
        prev_state = prev_state[[0, 2]]

        currrent_episode = []

        episode_return = 0
        rewards = []
        episode_len_count = 0

        hidden_state, cell_state = dqn_model.init_hidden_states(bsize=1)

        while episode_len_count < episode_max_len:
            
            episode_len_count += 1
            t_total_steps += 1
            
            epsilon = exploration.value(t_total_steps)

            if np.random.rand() < epsilon:
                torch_x = torch.from_numpy(prev_state).float().to(device)
                model_out = dqn_model.forward(torch_x, 1, 1, hidden_state, cell_state)
                action = np.random.randint(0, env.action_space.n)
                hidden_state = model_out[1][0]
                cell_state = model_out[1][1]
                
            else:
                torch_x = torch.from_numpy(prev_state).float().to(device)
                model_out = dqn_model.forward(torch_x, 1, 1, hidden_state, cell_state)
                out = model_out[0]
                action = int(torch.argmax(out[0]))
                hidden_state = model_out[1][0]
                cell_state = model_out[1][1]
            
            next_state, reward, done, _ = env.step(action)
            # env.render()
            next_state = next_state[[0, 2]]
            
            rewards.append(reward)
            
            currrent_episode.append((prev_state, action, reward, next_state, done))
            
            prev_state = next_state            
            
            # Copy weights to the target network
            if (t_total_steps % target_network_update_freq) == 0:
                dqn_target.load_state_dict(dqn_model.state_dict())
        
            # Training
            if (t_total_steps % train_freq) == 0:
                batch = replay_buffer.sample(batch_size, episode_training_len)
                loss = train_drqn_batch(optimizer, batch, episode_training_len, dqn_model, dqn_target, gamma)
                losses.append(loss)

            # Debugging
            if t_total_steps % print_freq == 0:
                logger.record_tabular("steps", t_total_steps)
                logger.record_tabular("episodes", t_episode)
                logger.record_tabular("short episodes", short_episode_cnt)
                logger.record_tabular("good episodes", good_episode_cnt)

                logger.record_tabular("mean reward", np.mean(last_100ep_returns))
                logger.record_tabular("mean len", np.mean(last_100ep_lens))

                logger.record_tabular("% time spent exploring", int(100 * epsilon))
                logger.dump_tabular()

            # Saving models and losses
            if t_total_steps % model_save_freq == 0:     
                torch.save(dqn_model.state_dict(),'model_{}.torch'.format(t_total_steps))

                with open("loss.pkl", "wb") as fp:
                    pickle.dump(losses, fp)    
                
            if done:
                t_episode += 1

                for i in range(len(rewards)):
                    episode_return += rewards[i] * pow(gamma, i)

                last_100ep_lens.append(len(currrent_episode))
                last_100ep_returns.append(episode_return)
                
                break

        if len(currrent_episode) >= episode_training_len:
            replay_buffer.add_episode(currrent_episode)
            good_episode_cnt += 1
        else:
            short_episode_cnt += 1