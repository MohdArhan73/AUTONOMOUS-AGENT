# reinforcement_learning/agent.py

import random
import torch


class Agent:
    """An individual agent controlled by a shared DQN model."""

    def __init__(
        self,
        agent_id,
        model,
        replay_buffer,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        device="cpu"
    ):
        self.agent_id = agent_id
        self.model = model
        self.replay_buffer = replay_buffer

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device(device)

        self.position = None
        self.path = []
        self.total_reward = 0.0
        self.steps = 0

        self.success = False
        self.dead = False

    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy exploration.

        During training, the agent explores randomly with
        probability epsilon. Otherwise, it selects the action
        with the highest predicted Q-value.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(4)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """Save an experience in the shared replay buffer."""
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

    def reset(self, start_position):
        """Reset the agent for a new episode."""
        self.position = start_position
        self.path = [start_position]

        self.total_reward = 0.0
        self.steps = 0

        self.success = False
        self.dead = False

    def update_after_step(self, position, reward, done, success=False):
        """Update the agent's episode statistics."""
        self.position = position
        self.path.append(position)

        self.total_reward += reward
        self.steps += 1

        if done:
            self.success = success
            self.dead = not success

    def decay_epsilon(self):
        """Reduce exploration as training progresses."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )