import numpy as np
from collections import deque
import random

class RecurrentReplayMemory:
    def __init__(self, max_size, episode_min_len, episode_max_len):
        """Replay memory implemented as a queue.

        Args:
            - max_size: Maximum size of the buffer.
            - episode_min_len: minimum length of an eligible episode
            - episode_max_len: maximimum length of an eligible episode
        """
        self.max_size = max_size
        self.memory = deque(maxlen = self.max_size)
        self.episode_min_len = episode_min_len
        self.episode_max_len = episode_max_len

    def add_episode(self, episode):
        """Add an episode to the buffer.

        :param episode:  episode to add.
        """
        assert len(episode) >= self.episode_min_len
        self.memory.append(episode)

    def pre_populate(self, env, replay_prepopulate_steps):
        """Prepopulate the replay buffer before training.

        Args:
            - env: Environment to run
            - replay_prepopulate_steps: How many steps to pre-populated
        """

        episode_cnt = 0
        while episode_cnt < replay_prepopulate_steps:
            
            state = env.reset()
            state = state[[0, 2]]

            step_count = 0
            episode = []
            
            while step_count < self.episode_max_len:
                
                step_count +=1
                action = np.random.randint(0, env.action_space.n)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state[[0, 2]]

                episode.append((state, action, reward, next_state, done))

                if done:
                    break
                
                state = next_state
            
            if (len(episode) > self.episode_min_len):
                self.add_episode(episode)
                episode_cnt += 1
                
        print('Done pre-populated with %d episodes'%(len(self.memory)))         
        

    def sample(self, batch_size, episode_len):
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size:  Number of transitions to sample.
        :rtype: Batch (list)
        """

        sampled_episodes = random.sample(self.memory, batch_size)
        batch = []
        for episode in sampled_episodes:
            if len(episode) + 1 - episode_len > 0:
                point = np.random.randint(0, len(episode) + 1 - episode_len)
                batch.append(episode[point:point + episode_len])

        assert len(batch) == batch_size
        return batch