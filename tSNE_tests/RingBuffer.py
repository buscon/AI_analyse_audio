import numpy as np

class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) %self.data.size
        return self.data[idx]

def ringbuff_numpy_test():
    ringlen = 100000
    ringbuff = RingBuffer(ringlen)
    for i in range(40):
        ringbuff.extend(np.random.rand(10000)) # write
        print(ringbuff.get() )#read


ringbuff_numpy_test()
