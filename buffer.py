from common_definitions import *
import random
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=int(buffer_size))  # (s,a,r,s')

        # constant sizes
        self.batch_size = batch_size

        # temp variable
        self.p_indices = [BUFFER_UNBALANCE_GAP/2]

    def append(self, s, a, r, sn, d):
        self.buffer.append([s, a, np.expand_dims(r, -1), sn, np.expand_dims(d, -1)])

    def get_batch(self, unbalance_p=True):
        # unbalance indices
        p_indices = None
        if random.random() < unbalance_p:
            # self.p_indices.extend(np.log2(np.array(range(len(self.p_indices), len(self.buffer)))+2))
            self.p_indices.extend((np.arange(len(self.buffer)-len(self.p_indices))+1)*BUFFER_UNBALANCE_GAP+self.p_indices[-1])
            p_indices = self.p_indices / np.sum(self.p_indices)

        chosen_indices = np.random.choice(len(self.buffer),
                                          size=min(self.batch_size, len(self.buffer)),
                                          replace=False,
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
