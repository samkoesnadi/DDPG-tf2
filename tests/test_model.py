import unittest
import numpy as np
from model_utils import generate_noise, reset_noise, generate_action, store_memory, update_networks

class TestModelUtils(unittest.TestCase):

    def test_generate_noise(self):
        mean = np.zeros(1)
        std_deviation = 0.1
        noise = generate_noise(mean, std_deviation)
        self.assertEqual(noise.shape, mean.shape)
        self.assertIsInstance(noise, np.ndarray)

    def test_reset_noise(self):
        mean = np.zeros(1)
        std_deviation = 0.1
        reset_noise = reset_noise(mean, std_deviation)
        self.assertEqual(reset_noise.shape, mean.shape)
        self.assertIsInstance(reset_noise, np.ndarray)

    def test_generate_action(self):
        state = np.array([0.1, 0.2, 0.3])
        action = generate_action(state)
        self.assertTrue(-1 <= action <= 1)

    def test_store_memory(self):
        prev_state = np.array([0.1, 0.2, 0.3])
        action = 0.5
        reward = 1.0
        state = np.array([0.4, 0.5, 0.6])
        done = False
        buffer = store_memory(prev_state, action, reward, state, done)
        self.assertIn((prev_state, action, reward, state, done), buffer)

    def test_update_networks(self):
        states = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        actions = np.array([0.5, -0.5])
        rewards = np.array([1.0, -1.0])
        next_states = np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
        done_values = np.array([False, True])
        critic_loss, actor_loss = update_networks(states, actions, rewards, next_states, done_values)
        self.assertIsInstance(critic_loss, float)
        self.assertIsInstance(actor_loss, float)

if __name__ == '__main__':
    unittest.main()
