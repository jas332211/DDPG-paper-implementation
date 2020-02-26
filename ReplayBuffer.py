from collections import deque,namedtuple
import gym
import random

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward','terminal')
                        )



class ReplayBuffer(object):

    def __init__(self, size):
        self.buffer = []
        self.size = size
        self.position = 0
        #for i in range size: 
         #   self.buffer.append(None)

    def push(self, *args):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)