"""
Common definitions of variables that can be used across files
"""

try:
    from tensorflow.keras.initializers import glorot_normal  # pylint: disable=no-name-in-module
except ImportError:
    print("TensorFlow could not be imported. Please ensure that TensorFlow is installed and the version is compatible.")

# brain parameters
GAMMA = 0.99  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = glorot_normal()

# buffer params
UNBALANCE_P = 0.8  # newer entries are prioritized
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 200
BUFFER_SIZE = 1e6
TOTAL_EPISODES = 10000
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4
WARM_UP = 1  # num of warm up epochs
