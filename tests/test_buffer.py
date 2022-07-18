"""
Test ReplayBuffer class
"""

import numpy as np

from src.buffer import ReplayBuffer


def test_replay_buffer():
    """
    It takes in a state, an action, a reward,
    a next state, and a done flag, and adds it to the buffer.
    """
    buffer = ReplayBuffer(1e6, 200)
    buffer.append(2.0, 0.95, 1.0, 2.5, int(False))
    assert buffer.get_batch(False) == [[2.0, 0.95, np.array([1.]), 2.5, np.array([0])]]
