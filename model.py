"""
The main model declaration
"""
import logging
import os

import numpy as np
import tensorflow as tf

from common_definitions import (KERNEL_INITIALIZER, GAMMA, RHO, STD_DEV, BUFFER_SIZE, BATCH_SIZE,
                                CRITIC_LR, ACTOR_LR)
from buffer import ReplayBuffer
from utils import OUActionNoise


def ActorNetwork(num_states=24, num_actions=4, action_high=1):
    """
    Get Actor Network with the given parameters.

    Args:
        num_states: number of states in the NN
        num_actions: number of actions in the NN
        action_high: the top value of the action

    Returns:
        the Keras Model
    """
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_normal_initializer(stddev=0.0005)

    inputs = tf.keras.layers.Input(shape=(num_states,), dtype=tf.float32)
    out = tf.keras.layers.Dense(600, activation=tf.nn.leaky_relu,
                                kernel_initializer=KERNEL_INITIALIZER)(inputs)
    out = tf.keras.layers.Dense(300, activation=tf.nn.leaky_relu,
                                kernel_initializer=KERNEL_INITIALIZER)(out)
    outputs = tf.keras.layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(
        out) * action_high

    model = tf.keras.Model(inputs, outputs)
    return model


def CriticNetwork(num_states=24, num_actions=4, action_high=1):
    """
    Get Critic Network with the given parameters.

    Args:
        num_states: number of states in the NN
        num_actions: number of actions in the NN
        action_high: the top value of the action

    Returns:
        the Keras Model
    """
    last_init = tf.random_normal_initializer(stddev=0.00005)

    # State as input
    state_input = tf.keras.layers.Input(shape=(num_states), dtype=tf.float32)
    state_out = tf.keras.layers.Dense(600, activation=tf.nn.leaky_relu,
                                      kernel_initializer=KERNEL_INITIALIZER)(state_input)
    state_out = tf.keras.layers.BatchNormalization()(state_out)
    state_out = tf.keras.layers.Dense(300, activation=tf.nn.leaky_relu,
                                      kernel_initializer=KERNEL_INITIALIZER)(state_out)

    # Action as input
    action_input = tf.keras.layers.Input(shape=(num_actions), dtype=tf.float32)
    action_out = tf.keras.layers.Dense(300, activation=tf.nn.leaky_relu,
                                       kernel_initializer=KERNEL_INITIALIZER)(
        action_input / action_high)

    # Both are passed through seperate layer before concatenating
    added = tf.keras.layers.Add()([state_out, action_out])

    added = tf.keras.layers.BatchNormalization()(added)
    outs = tf.keras.layers.Dense(150, activation=tf.nn.leaky_relu,
                                 kernel_initializer=KERNEL_INITIALIZER)(added)
    outs = tf.keras.layers.BatchNormalization()(outs)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(outs)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def update_target(model_target, model_ref, rho=0):
    """
    Update target's weights with the given model reference

    Args:
        model_target: the target model to be changed
        model_ref: the reference model
        rho: the ratio of the new and old weights
    """
    model_target.set_weights([rho * ref_weight + (1 - rho) * target_weight
                              for (target_weight, ref_weight) in
                              list(zip(model_target.get_weights(), model_ref.get_weights()))])


class Brain:
    """
    The Brain that contains all the models
    """

    def __init__(self, num_states, num_actions, action_high, action_low, gamma=GAMMA, rho=RHO,
                 std_dev=STD_DEV):
        # initialize everything
        self.actor_network = ActorNetwork(num_states, num_actions, action_high)
        self.critic_network = CriticNetwork(num_states, num_actions, action_high)
        self.actor_target = ActorNetwork(num_states, num_actions, action_high)
        self.critic_target = CriticNetwork(num_states, num_actions, action_high)

        # Making the weights equal initially
        self.actor_target.set_weights(self.actor_network.get_weights())
        self.critic_target.set_weights(self.critic_network.get_weights())

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.gamma = tf.constant(gamma)
        self.rho = rho
        self.action_high = action_high
        self.action_low = action_low
        self.num_states = num_states
        self.num_actions = num_actions
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR, amsgrad=True)
        self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR, amsgrad=True)

        # temporary variable for side effects
        self.cur_action = None

        # define update weights with tf.function for improved performance
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, num_states), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_actions), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_states), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ])
        def update_weights(s, a, r, sn, d):
            """
            Function to update weights with optimizer
            """
            with tf.GradientTape() as tape:
                # define target
                y = r + self.gamma * (1 - d) * self.critic_target([sn, self.actor_target(sn)])
                # define the delta Q
                critic_loss = tf.math.reduce_mean(tf.math.abs(y - self.critic_network([s, a])))
            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables))

            with tf.GradientTape() as tape:
                # define the delta mu
                actor_loss = -tf.math.reduce_mean(self.critic_network([s, self.actor_network(s)]))
            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_network.trainable_variables))
            return critic_loss, actor_loss

        self.update_weights = update_weights

    def act(self, state, _notrandom=True, noise=True):
        """
        Run action by the actor network

        Args:
            state: the current state
            _notrandom: whether greedy is used
            noise: whether noise is to be added to the result action (this improves exploration)

        Returns:
            the resulting action
        """
        self.cur_action = (self.actor_network(state)[0].numpy()
                           if _notrandom
                           else (np.random.uniform(self.action_low, self.action_high,
                                                   self.num_actions)) +
                                (self.noise() if noise else 0))
        self.cur_action = np.clip(self.cur_action, self.action_low, self.action_high)

        return self.cur_action

    def remember(self, prev_state, reward, state, done):
        """
        Store states, reward, done value to the buffer
        """
        # record it in the buffer based on its reward
        self.buffer.append(prev_state, self.cur_action, reward, state, done)

    def learn(self, entry):
        """
        Run update for all networks (for training)
        """
        s, a, r, sn, d = zip(*entry)

        c_l, a_l = self.update_weights(tf.convert_to_tensor(s, dtype=tf.float32),
                                       tf.convert_to_tensor(a, dtype=tf.float32),
                                       tf.convert_to_tensor(r, dtype=tf.float32),
                                       tf.convert_to_tensor(sn, dtype=tf.float32),
                                       tf.convert_to_tensor(d, dtype=tf.float32))

        update_target(self.actor_target, self.actor_network, self.rho)
        update_target(self.critic_target, self.critic_network, self.rho)

        return c_l, a_l

    def save_weights(self, path):
        """
        Save weights to `path`
        """
        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        self.actor_network.save_weights(path + "an.h5")
        self.critic_network.save_weights(path + "cn.h5")
        self.critic_target.save_weights(path + "ct.h5")
        self.actor_target.save_weights(path + "at.h5")

    def load_weights(self, path):
        """
        Load weights from path
        """
        try:
            self.actor_network.load_weights(path + "an.h5")
            self.critic_network.load_weights(path + "cn.h5")
            self.critic_target.load_weights(path + "ct.h5")
            self.actor_target.load_weights(path + "at.h5")
        except OSError as err:
            logging.warning("Weights files cannot be found, %s", err)
