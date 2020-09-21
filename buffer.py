from common_definitions import *
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque()  # (s,a,r,s')

        # constant sizes
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def append(self, s, a, r, sn):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.popleft()

        self.buffer.append([s, a, np.expand_dims(r, -1), sn])

    def get_batch(self, unbalance_p=True):
        # unbalance indices
        p_indices = None
        if unbalance_p:
            p_indices = np.log10(np.array(range(len(self.buffer)))+2)
            p_indices /= np.sum(p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          p=p_indices)

        # # sort it
        # chosen_indices.sort()
        #
        # # run the iteration
        # buffer = [self.buffer.pop(chosen_index-i_c) for i_c, chosen_index in enumerate(chosen_indices)]

        buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

        return buffer


if __name__ == "__main__":
    rb = ReplayBuffer(10, 5)

    for i in range(100):
        rb.append(1,2,3,4,5)
        if i % 10 == 0:
            (rb.get_batch())

        print(len(rb.buffer))
