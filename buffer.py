from common_definitions import *
import random

class ReplayBuffer():
    def __init__(self, buffer_size, batch_size):
        self.buffer = list()  # (s,a,r,s',d)

        # constant sizes
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def append(self, s, a, r, sn, d):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)

        self.buffer.append([s, a, r, sn, d])

    def get_batch(self, unbalance_p=True):
        if len(self.buffer) < self.batch_size:
            # buffer = self.buffer.copy()
            # self.buffer.clear()

            raise Exception("Buffer size insufficient: available", len(self.buffer))
        else:
            # unbalance indices
            p_indices = None
            if unbalance_p:
                p_indices = np.log2(np.array(range(len(self.buffer)))+1.01)
                p_indices /= np.sum(p_indices)

            chosen_indices = np.random.choice(len(self.buffer),
                                              size=self.batch_size,
                                              replace=False,
                                              p=p_indices)

            # # sort it
            # chosen_indices.sort()
            #
            # # run the iteration
            # buffer = [self.buffer.pop(chosen_index-i_c) for i_c, chosen_index in enumerate(chosen_indices)]

            buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]
        # print(len(self.buffser))
        return buffer


if __name__ == "__main__":
    rb = ReplayBuffer(10, 5)

    for i in range(100):
        rb.append(1,2,3,4,5)
        if i % 10 == 0:
            (rb.get_batch())

        print(len(rb.buffer))
