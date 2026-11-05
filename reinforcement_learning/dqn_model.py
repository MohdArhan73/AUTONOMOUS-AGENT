# reinforcement_learning/dqn_model.py

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network for grid-based pathfinding.

    Input:
        8 state features.

    Output:
        4 Q-values corresponding to the four possible actions.
    """

    def __init__(self, state_size=8, action_size=4, hidden_size=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        """Return predicted Q-values for the supplied state."""
        return self.network(state)