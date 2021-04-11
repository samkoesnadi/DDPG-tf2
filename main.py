"""
Main file
"""
import argparse
import logging
import random

import gym
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from common_definitions import CHECKPOINTS_PATH, TOTAL_EPISODES, TF_LOG_DIR, UNBALANCE_P
from model import Brain
from utils import Tensorboard


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        prog="Deep Deterministic Policy Gradient (DDPG)",
        description="Deep Deterministic Policy Gradient (DDPG) in Tensorflow 2"
    )
    parser.add_argument('--env', type=str, nargs='?',
                        default="BipedalWalker-v3",
                        help='The OpenAI Gym environment to train on, '
                             'e.g. BipedalWalker-v3, LunarLanderContinuous-v2,'
                             ' Pendulum-v0')
    parser.add_argument('--render_env', type=bool, nargs='?', default=True,
                        help='Render the environment to be visually visible')
    parser.add_argument('--train', type=bool, nargs='?', default=True,
                        help='Train the network on the modified DDPG algorithm')
    parser.add_argument('--use_noise', type=bool, nargs='?', default=True,
                        help='OU Noise will be applied to the policy action')
    parser.add_argument('--eps_greedy', type=float, nargs='?', default=0.95,
                        help="The epsilon for Epsilon-greedy in the policy's action")
    parser.add_argument('--warm_up', type=bool, nargs='?', default=1,
                        help='Following recommendation from OpenAI Spinning Up, the actions in the '
                             'early epochs can be set random to increase exploration. This warm up '
                             'defines how many epochs are initially set to do this.')
    parser.add_argument('--save_weights', type=bool, nargs='?', default=True,
                        help='Save the weight of the network in the defined checkpoint file '
                             'directory.')

    args = parser.parse_args()
    RL_TASK = args.env
    RENDER_ENV = args.render_env
    LEARN = args.train
    USE_NOISE = args.use_noise
    WARM_UP = args.warm_up
    SAVE_WEIGHTS = args.save_weights
    EPS_GREEDY = args.eps_greedy

    # Step 1. create the gym environment
    env = gym.make(RL_TASK)
    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]

    brain = Brain(env.observation_space.shape[0], env.action_space.shape[0], action_space_high,
                  action_space_low)
    tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

    # load weights if available
    logging.info("Loading weights from %s*, make sure the folder exists", CHECKPOINTS_PATH)
    brain.load_weights(CHECKPOINTS_PATH)

    # all the metrics
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # run iteration
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            prev_state = env.reset()
            acc_reward.reset_states()
            actions_squared.reset_states()
            Q_loss.reset_states()
            A_loss.reset_states()
            brain.noise.reset()

            for _ in range(2000):
                if RENDER_ENV:  # render the environment into GUI
                    env.render()

                # Recieve state and reward from environment.
                cur_act = brain.act(tf.expand_dims(prev_state, 0), _notrandom=(ep >= WARM_UP) and
                                    (random.random() < EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES),
                                    noise=USE_NOISE)
                state, reward, done, _ = env.step(cur_act)
                brain.remember(prev_state, reward, state, int(done))

                # update weights
                if LEARN:
                    c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
                    Q_loss(c)
                    A_loss(a)

                # post update for next step
                acc_reward(reward)
                actions_squared(np.square(cur_act/action_space_high))
                prev_state = state

                if done:
                    break

            ep_reward_list.append(acc_reward.result().numpy())
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            avg_reward_list.append(avg_reward)

            # print the average reward
            t.set_postfix(r=avg_reward)
            tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

            # save weights
            if ep % 5 == 0 and SAVE_WEIGHTS:
                brain.save_weights(CHECKPOINTS_PATH)

    env.close()
    brain.save_weights(CHECKPOINTS_PATH)

    logging.info("Training done...")

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
