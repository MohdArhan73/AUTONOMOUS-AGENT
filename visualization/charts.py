import os
import matplotlib.pyplot as plt


class TrainingCharts:
    def __init__(self, output_dir="data/charts"):
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def plot_rewards(self, history, save=True, show=True):
        """Plot average reward across generations."""
        if not history:
            print("No training history available.")
            return

        generations = [
            item.get("generation", index + 1)
            for index, item in enumerate(history)
        ]

        rewards = [
            item.get("average_reward", 0)
            for item in history
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(generations, rewards, label="Average Reward")

        plt.title("DQN Training Rewards")
        plt.xlabel("Generation")
        plt.ylabel("Average Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        self._finish_plot("training_rewards.png", save, show)

    def plot_success_rate(self, history, save=True, show=True):
        """Plot success rate across generations."""
        if not history:
            print("No training history available.")
            return

        generations = [
            item.get("generation", index + 1)
            for index, item in enumerate(history)
        ]

        success_rates = [
            item.get("success_rate", 0)
            for item in history
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(
            generations,
            success_rates,
            label="Success Rate"
        )

        plt.title("Agent Success Rate")
        plt.xlabel("Generation")
        plt.ylabel("Success Rate (%)")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        self._finish_plot("success_rate.png", save, show)

    def plot_path_lengths(
        self,
        dqn_lengths,
        astar_lengths,
        save=True,
        show=True
    ):
        """Compare DQN and A* path lengths."""
        if not dqn_lengths or not astar_lengths:
            print("Path-length comparison data is incomplete.")
            return

        dqn_mean = sum(dqn_lengths) / len(dqn_lengths)
        astar_mean = sum(astar_lengths) / len(astar_lengths)

        labels = ["DQN", "A*"]
        lengths = [dqn_mean, astar_mean]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, lengths)

        plt.title("Average Path Length Comparison")
        plt.ylabel("Number of Steps")
        plt.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, lengths):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()

        self._finish_plot("path_length_comparison.png", save, show)

    def plot_training_loss(self, losses, save=True, show=True):
        """Plot neural network training loss."""
        if not losses:
            print("No loss history available.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(losses) + 1), losses)

        plt.title("DQN Training Loss")
        plt.xlabel("Training Update")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self._finish_plot("training_loss.png", save, show)

    def plot_all(self, history, save=True, show=True):
        """Generate the reward and success-rate charts."""
        self.plot_rewards(
            history,
            save=save,
            show=show
        )

        self.plot_success_rate(
            history,
            save=save,
            show=show
        )

    def _finish_plot(self, filename, save, show):
        """Save and/or display the current Matplotlib figure."""
        if save:
            filepath = os.path.join(
                self.output_dir,
                filename
            )

            plt.savefig(filepath, dpi=150)
            print(f"Chart saved: {filepath}")

        if show:
            plt.show()

        plt.close()