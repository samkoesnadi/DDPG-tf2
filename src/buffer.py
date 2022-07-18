"""
Buffer system for the RL
"""

import random
from collections import deque

import numpy as np

from src.common_definitions import BUFFER_UNBALANCE_GAP


class ReplayBuffer:
    """
    Replay Buffer to store the experiences.
    """

    def __init__(self, buffer_size, batch_size):
        """
        Initialize the attributes.

        Args:
            buffer_size: The size of the buffer memory
            batch_size: The batch for each of the data request `get_batch`
        """
        self.buffer = deque(maxlen=int(buffer_size))  # with format of (s,a,r,s')

        # constant sizes to use
        self.batch_size = batch_size

        # temp variables
        self.p_indices = [BUFFER_UNBALANCE_GAP/2]

    def append(self, state, action, reward, next_state, done):  # pylint: disable=too-many-arguments
        """
        Append to the Buffer

        Args:
            state: the state
            action: the action
            r: the reward
            sn: the next state
            d: done (whether one loop is done or not)
        """
        self.buffer.append([
            state, action, np.expand_dims(reward, -1),
            next_state, np.expand_dims(done, -1)
        ])

    def get_batch(self, unbalance_p=True):
        """
        Get the batch randomly from the buffer

        Args:
            unbalance_p: If true, unbalance probability of taking the batch from buffer with
            recent event being more prioritized

        Returns:
            the resulting batch
        """
        # unbalance indices
        p_indices = None
        if random.random() < unbalance_p:
            self.p_indices.extend((np.arange(len(self.buffer)-len(self.p_indices))+1)
                                  * BUFFER_UNBALANCE_GAP + self.p_indices[-1])
            p_indices = self.p_indices / np.sum(self.p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False,
                                          p=p_indices)

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

        return buffer
