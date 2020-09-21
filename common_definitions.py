import tensorflow as tf
import numpy as np


# general parameters
RL_TASK = 'BipedalWalker-v2'
RENDER_ENV = True
CHECKPOINTS_PATH = "./checkpoints/DDPG_"
TF_LOG_DIR = './logs/DDPG/'

# brain parameters
GAMMA = 0.99  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = tf.keras.initializers.he_uniform()

# buffer params
UNBALANCE_P = True  # newer entries are prioritized

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 64
BUFFER_SIZE = 1e6
TOTAL_EPISODES = 100000
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4
DROUPUT_N = 0.1
WARM_UP = 3  # num of warm up epochs