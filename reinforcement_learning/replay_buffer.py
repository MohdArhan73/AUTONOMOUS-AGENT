# reinforcement_learning/replay_buffer.py

import random
from collections import deque


class ReplayBuffer:
    """Stores and samples experiences for DQN training."""

    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Store one experience."""
        experience = (
            list(state),
            int(action),
            float(reward),
            list(next_state),
            bool(done)
        )

        self.memory.append(experience)

    def sample(self, batch_size):
        """Return a random batch of experiences."""
        if batch_size > len(self.memory):
            raise ValueError(
                "Batch size cannot exceed the number of stored experiences."
            )

        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the number of stored experiences."""
        return len(self.memory)

    def clear(self):
        """Remove all stored experiences."""
        self.memory.clear()