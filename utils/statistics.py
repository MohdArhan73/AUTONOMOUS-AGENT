import csv
import os
import time


class Statistics:
    def __init__(self, output_file=None):
        self.output_file = output_file

        self.generation_history = []
        self.episode_history = []
        self.loss_history = []

        self.dqn_path_lengths = []
        self.astar_path_lengths = []

        self.training_start_time = None
        self.training_end_time = None

    def start_training(self):
        """Record the start time of training."""
        self.training_start_time = time.time()
        self.training_end_time = None

    def stop_training(self):
        """Record the end time of training."""
        self.training_end_time = time.time()

    def record_episode(
        self,
        agent_id,
        reward,
        steps,
        success,
        generation=None
    ):
        """Record the results of one agent episode."""
        episode_data = {
            "agent_id": agent_id,
            "generation": generation,
            "reward": float(reward),
            "steps": int(steps),
            "success": bool(success)
        }

        self.episode_history.append(episode_data)

    def record_generation(
        self,
        generation,
        rewards,
        successes,
        path_lengths=None,
        losses=None
    ):
        """Calculate and store metrics for one generation."""
        rewards = list(rewards)
        successes = list(successes)
        path_lengths = list(path_lengths or [])
        losses = list(losses or [])

        average_reward = (
            sum(rewards) / len(rewards)
            if rewards else 0.0
        )

        success_count = sum(bool(value) for value in successes)

        success_rate = (
            (success_count / len(successes)) * 100
            if successes else 0.0
        )

        average_path_length = (
            sum(path_lengths) / len(path_lengths)
            if path_lengths else 0.0
        )

        average_loss = (
            sum(losses) / len(losses)
            if losses else 0.0
        )

        record = {
            "generation": int(generation),
            "average_reward": round(average_reward, 4),
            "success_rate": round(success_rate, 2),
            "successful_agents": success_count,
            "total_agents": len(successes),
            "average_path_length": round(
                average_path_length, 2
            ),
            "average_loss": round(average_loss, 6)
        }

        self.generation_history.append(record)

        return record

    def record_loss(self, loss):
        """Store a neural network training loss."""
        self.loss_history.append(float(loss))

    def record_path_comparison(self, dqn_length, astar_length):
        """Store successful DQN and A* path lengths."""
        if dqn_length is not None and dqn_length >= 0:
            self.dqn_path_lengths.append(int(dqn_length))

        if astar_length is not None and astar_length >= 0:
            self.astar_path_lengths.append(int(astar_length))

    def get_training_duration(self):
        """Return the elapsed training time in seconds."""
        if self.training_start_time is None:
            return 0.0

        end_time = (
            self.training_end_time
            if self.training_end_time is not None
            else time.time()
        )

        return max(0.0, end_time - self.training_start_time)

    def get_summary(self):
        """Return an overall summary of training performance."""
        total_episodes = len(self.episode_history)

        successful_episodes = sum(
            episode["success"]
            for episode in self.episode_history
        )

        failed_episodes = total_episodes - successful_episodes

        rewards = [
            episode["reward"]
            for episode in self.episode_history
        ]

        successful_steps = [
            episode["steps"]
            for episode in self.episode_history
            if episode["success"]
        ]

        return {
            "total_episodes": total_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "overall_success_rate": round(
                successful_episodes / total_episodes * 100,
                2
            ) if total_episodes else 0.0,
            "overall_average_reward": round(
                sum(rewards) / len(rewards),
                4
            ) if rewards else 0.0,
            "average_successful_path_length": round(
                sum(successful_steps) / len(successful_steps),
                2
            ) if successful_steps else 0.0,
            "training_duration_seconds": round(
                self.get_training_duration(),
                2
            )
        }

    def save_generation_history(self, filepath=None):
        """Save generation-level metrics to a CSV file."""
        filepath = filepath or self.output_file

        if not filepath:
            raise ValueError("A CSV output filepath is required.")

        os.makedirs(
            os.path.dirname(os.path.abspath(filepath)),
            exist_ok=True
        )

        fieldnames = [
            "generation",
            "average_reward",
            "success_rate",
            "successful_agents",
            "total_agents",
            "average_path_length",
            "average_loss"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.generation_history)

        print(f"Training history saved to: {filepath}")

    def save_episode_history(self, filepath):
        """Save individual episode results to a CSV file."""
        if not self.episode_history:
            print("No episode history available.")
            return

        os.makedirs(
            os.path.dirname(os.path.abspath(filepath)),
            exist_ok=True
        )

        fieldnames = [
            "agent_id",
            "generation",
            "reward",
            "steps",
            "success"
        ]

        with open(filepath, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(self.episode_history)

        print(f"Episode history saved to: {filepath}")

    def print_summary(self):
        """Print a readable training summary."""
        summary = self.get_summary()

        print("\n========== TRAINING SUMMARY ==========")

        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("======================================\n")

    def reset(self):
        """Clear all recorded metrics."""
        self.generation_history.clear()
        self.episode_history.clear()
        self.loss_history.clear()
        self.dqn_path_lengths.clear()
        self.astar_path_lengths.clear()

        self.training_start_time = None
        self.training_end_time = None