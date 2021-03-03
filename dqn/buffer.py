from collections import namedtuple, deque
import numpy as np
import random
print(__name__)

from config import BUFFER_SIZE, BATCH_SIZE


class ReplayBuffer:
    """Stores experiences from agent-environment interactions and enables this experiences to be sampled"""
    experience = namedtuple("experience", ["state", "action", "reward", "next_state", "done"])

    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self.memory = []
        self.pointer = 0
        self.buffer_size = buffer_size

    def add(self, state:np.ndarray, action:np.ndarray, reward:np.ndarray, next_state:np.ndarray, done:np.ndarray):
        """Adds an experience to the buffer memory"""
        try:
            self.memory[self.pointer] = self.experience(state, action, reward, next_state, done)
        except IndexError:
            self.memory.append(self.experience(state, action, reward, next_state, done))
        assert len(self.memory) <= self.buffer_size
        self.pointer = (self.pointer + 1) % self.buffer_size

    def sample(self, num_batch: int = BATCH_SIZE):
        """Samples experiences from memory. num_batch determines how many experiences are sampled"""
        samples = random.sample(self.memory, k=num_batch)
        state = np.array([sample.state for sample in samples], dtype=np.float32)
        action = np.array([sample.action for sample in samples], dtype=np.int32)
        reward = np.array([sample.reward for sample in samples], dtype=np.float32)
        next_state = np.array([sample.next_state for sample in samples], dtype=np.float32)
        done = np.array([sample.done for sample in samples], dtype=np.int32)
        return state, action, reward, next_state, done

    def __len__(self):
        """Number of experiences currently in memory"""
        return len(self.memory)
