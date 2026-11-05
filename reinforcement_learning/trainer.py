# reinforcement_learning/trainer.py

import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn_model import DQN
from .agent import Agent
from .replay_buffer import ReplayBuffer


class Trainer:
    """Manages shared DQN training across multiple agents."""

    def __init__(
        self,
        environment,
        num_agents=20,
        max_generations=500,
        batch_size=64,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        replay_capacity=50000,
        target_update_frequency=10,
        training_updates=100,
        seed=None
    ):
        self.environment = environment

        self.num_agents = num_agents
        self.max_generations = max_generations
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_frequency = target_update_frequency
        self.training_updates = training_updates

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Select the available computing device.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Shared online and target networks.
        self.model = DQN().to(self.device)
        self.target_model = copy.deepcopy(self.model).to(self.device)

        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

        # One replay buffer shared by all agents.
        self.replay_buffer = ReplayBuffer(replay_capacity)

        self.generation = 0
        self.total_agents_generated = 0
        self.total_successes = 0

        self.best_path = None
        self.best_path_length = float("inf")
        self.best_path_generation = None

        self.history = []

        self.paused = False
        self.stop_requested = False

        # Every agent uses the same model and replay buffer.
        self.agents = [
            Agent(
                agent_id=i + 1,
                model=self.model,
                replay_buffer=self.replay_buffer,
                epsilon=self.epsilon,
                epsilon_min=self.epsilon_min,
                epsilon_decay=self.epsilon_decay,
                device=self.device
            )
            for i in range(self.num_agents)
        ]

    def train_step(self):
        """Perform one DQN update using replayed experiences."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # Predicted Q-values for the selected actions.
        current_q_values = self.model(states).gather(
            1, actions
        ).squeeze(1)

        # Target Q-values from the target network.
        with torch.no_grad():
            next_q_values = self.target_model(
                next_states
            ).max(dim=1)[0]

            expected_q_values = rewards + (
                self.gamma * next_q_values * (1.0 - dones)
            )

        loss = self.loss_function(
            current_q_values,
            expected_q_values
        )

        self.optimizer.zero_grad()
        loss.backward()

        # Prevent excessively large gradients.
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=10.0
        )

        self.optimizer.step()

        return loss.item()

    def run_episode(self, agent, training=True):
        """Run one agent through a complete episode."""
        state = self.environment.reset()

        agent.reset(self.environment.agent_position)

        total_reward = 0.0
        episode_steps = 0

        while not self.environment.done:
            if self.stop_requested:
                break

            # Select an action.
            action = agent.select_action(
                state,
                training=training
            )

            # Perform the action in the environment.
            next_state, reward, done, info = (
                self.environment.step(action)
            )

            # Store the experience.
            if training:
                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

            agent.update_after_step(
                position=info["position"],
                reward=reward,
                done=done,
                success=info["success"]
            )

            state = next_state
            total_reward += reward
            episode_steps += 1

        return {
            "agent_id": agent.agent_id,
            "reward": total_reward,
            "steps": episode_steps,
            "success": self.environment.success,
            "path": list(agent.path),
            "path_length": max(0, len(agent.path) - 1)
        }

    def run_generation(self):
        """Run one generation and update the shared DQN."""
        if self.generation >= self.max_generations:
            return None

        if self.stop_requested:
            return None

        self.generation += 1

        generation_results = []
        start_time = time.perf_counter()

        # Run all agents using the shared policy.
        for agent in self.agents:
            if self.stop_requested:
                break

            result = self.run_episode(
                agent,
                training=True
            )

            generation_results.append(result)
            self.total_agents_generated += 1

            if result["success"]:
                self.total_successes += 1

                # Keep the shortest successful RL path.
                if result["path_length"] < self.best_path_length:
                    self.best_path_length = result["path_length"]
                    self.best_path = result["path"]
                    self.best_path_generation = self.generation

        if not generation_results:
            return None

        # Update the shared network after the generation.
        losses = []

        self.model.train()

        if len(self.replay_buffer) >= self.batch_size:
            for _ in range(self.training_updates):
                if self.stop_requested:
                    break

                loss = self.train_step()

                if loss is not None:
                    losses.append(loss)

        # Synchronize the target network periodically.
        if (
            self.generation % self.target_update_frequency == 0
        ):
            self.target_model.load_state_dict(
                self.model.state_dict()
            )

        # Decay epsilon once per generation.
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        for agent in self.agents:
            agent.epsilon = self.epsilon

        successful_agents = sum(
            result["success"]
            for result in generation_results
        )

        success_rate = (
            successful_agents / len(generation_results)
        )

        average_reward = float(np.mean([
            result["reward"]
            for result in generation_results
        ]))

        average_steps = float(np.mean([
            result["steps"]
            for result in generation_results
        ]))

        average_loss = (
            float(np.mean(losses))
            if losses else 0.0
        )

        duration = time.perf_counter() - start_time

        stats = {
            "generation": self.generation,
            "agents": len(generation_results),
            "successful_agents": successful_agents,
            "failed_agents": (
                len(generation_results) - successful_agents
            ),
            "success_rate": success_rate,
            "average_reward": average_reward,
            "average_steps": average_steps,
            "epsilon": self.epsilon,
            "best_path_length": (
                self.best_path_length
                if self.best_path is not None
                else None
            ),
            "average_loss": average_loss,
            "duration_seconds": duration
        }

        self.history.append(stats)

        return stats

    def train(self, generations=None):
        """Train for a specified number of generations."""
        if generations is None:
            generations = self.max_generations - self.generation

        results = []

        for _ in range(generations):
            if self.stop_requested:
                break

            if self.generation >= self.max_generations:
                break

            stats = self.run_generation()

            if stats is None:
                break

            results.append(stats)

        return results

    def pause(self):
        """Pause training between generations."""
        self.paused = True

    def resume(self):
        """Resume training."""
        self.paused = False

    def stop(self):
        """Request that the current training operation stop."""
        self.stop_requested = True

    def reset_training(self, reset_environment=False):
        """
        Reset the learning model and training statistics.

        If reset_environment is False, preserve the current maze.
        """
        if reset_environment:
            self.environment.reset_environment()
        else:
            self.environment.reset()

        self.model = DQN().to(self.device)

        self.target_model = copy.deepcopy(
            self.model
        ).to(self.device)

        self.target_model.eval()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

        self.replay_buffer.clear()

        self.generation = 0
        self.total_agents_generated = 0
        self.total_successes = 0

        self.best_path = None
        self.best_path_length = float("inf")
        self.best_path_generation = None

        self.history.clear()

        self.epsilon = 1.0
        self.paused = False
        self.stop_requested = False

        # Reconnect every agent to the new shared model.
        for agent in self.agents:
            agent.model = self.model
            agent.replay_buffer = self.replay_buffer
            agent.epsilon = self.epsilon

            agent.reset(self.environment.grid.start)

    def get_statistics(self):
        """Return the current training summary."""
        return {
            "generation": self.generation,
            "total_agents": self.total_agents_generated,
            "total_successes": self.total_successes,
            "overall_success_rate": (
                self.total_successes / self.total_agents_generated
                if self.total_agents_generated else 0.0
            ),
            "epsilon": self.epsilon,
            "best_path_length": (
                self.best_path_length
                if self.best_path is not None
                else None
            ),
            "best_path_generation": self.best_path_generation,
            "replay_buffer_size": len(self.replay_buffer)
        }