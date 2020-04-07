import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Arguments:
            capacity: Max number of elements in buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.__counter__ = 0

    def push(self, s0, a, s1, r, d):
        """Push an element to the buffer.
        Arguments:
            s0: State before action
            a: Action picked by the agent
            s1: State after performing the action
            r: Reward recieved is state s1.
            d: Whether the episode terminated after in the state s1.
        If the buffer is full, start to rewrite elements
        starting from the oldest ones.
        """
        
        state = np.array(s0).reshape((1, len(s0)))
        next_state = np.array(s1).reshape((1, len(s1)))
        
        if self.__counter__ >= self.capacity:
            self.buffer[self.__counter__ % (self.capacity)] = (state, a, next_state, r, d)
            
        else: 
            self.buffer.append((state, a, next_state, r, d))
        
        self.__counter__ += 1

    def sample(self, batch_size):
        """Return `batch_size` randomly chosen elements."""
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        """Return size of the buffer."""
        return len(self.buffer)