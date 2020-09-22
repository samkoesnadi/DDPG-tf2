import tensorflow as tf
import numpy as np


# general parameters
RL_TASK = 'BipedalWalker-v3'
# RL_TASK = 'LunarLanderContinuous-v2'
# RL_TASK = 'Pendulum-v0'
RENDER_ENV = False
CHECKPOINTS_PATH = "./checkpoints/DDPG_"
TF_LOG_DIR = './logs/DDPG/'
EPS_GREEDY = 0.95

# brain parameters
GAMMA = 0.99  # for the temporal difference
RHO = 0.0005  # to update the target networks
KERNEL_INITIALIZER = tf.keras.initializers.glorot_normal()

# buffer params
UNBALANCE_P = True  # newer entries are prioritized
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 64
BUFFER_SIZE = 1e5
TOTAL_EPISODES = 10000
CRITIC_LR = 5e-4
ACTOR_LR = 1e-4
DROUPUT_N = 0.2
WARM_UP = 3  # num of warm up epochs
SAVE_WEIGHTS = True