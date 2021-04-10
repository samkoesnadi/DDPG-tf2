from common_definitions import *
import random
from collections import deque


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

    def append(self, state, action, r, sn, d):
        self.buffer.append([state, action, np.expand_dims(r, -1), sn, np.expand_dims(d, -1)])

    def get_batch(self, unbalance_p=True):
        # unbalance indices
        p_indices = None
        if random.random() < unbalance_p:
            self.p_indices.extend((np.arange(len(self.buffer)-len(self.p_indices))+1)*BUFFER_UNBALANCE_GAP+self.p_indices[-1])
            p_indices = self.p_indices / np.sum(self.p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False,
                                          p=p_indices)

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

        return buffer


if __name__ == "__main__":
    rb = ReplayBuffer(10, 5)

    for i in range(100):
        rb.append(1,2,3,4,5)
        if i % 10 == 0:
            (rb.get_batch())

        print(len(rb.buffer))
