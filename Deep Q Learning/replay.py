from collections import namedtuple
import torch
import numpy as np

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
)


class ReplayMemory:
    def __init__(self, max_size, state_size):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer.
            - state_size: Size of the state-space features for the environment.
        """
        self.max_size = max_size
        self.state_size = state_size

        # preallocating all the required memory, for speed concerns
        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.float)

        # pointer to the current location in the circular buffer
        self.idx = 0
        # indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        :param state:  1-D np.ndarray of state-features.
        :param action:  integer action.
        :param reward:  float reward.
        :param next_state:  1-D np.ndarray of state-features.
        :param done:  boolean value indicating the end of an episode.
        """

        self.states[self.idx] = torch.tensor(state, dtype=torch.float)
        self.actions[self.idx] = torch.tensor(action, dtype=torch.float)
        self.rewards[self.idx] = torch.tensor(reward, dtype=torch.float)
        self.next_states[self.idx] = torch.tensor(next_state, dtype=torch.float)
        self.dones[self.idx] = torch.tensor(done, dtype=torch.float)

        # circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size:  Number of transitions to sample.
        :rtype: Batch
        """

        all_index = np.arange(self.size)
        np.random.shuffle(all_index)
        
        # If the buffer contains less that batch_size then sample all
        if self.size < batch_size:
            batch_size = self.size
            
        sample_indices = all_index[:batch_size]

        batch = Batch(self.states[sample_indices], self.actions[sample_indices], self.rewards[sample_indices],
                     self.next_states[sample_indices], self.dones[sample_indices])

        return batch