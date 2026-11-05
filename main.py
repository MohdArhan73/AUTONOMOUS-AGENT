import random
import os
import torch

from algorithms.astar import AStar
from environment.environment import PathfindingEnvironment
from reinforcement_learning.trainer import Trainer
from utils.configuration import Config
from utils.statistics import Statistics
from visualization.renderer import Renderer
from visualization.dashboard import Dashboard
from visualization.charts import TrainingCharts


def main():
    """Initialize and run the pathfinding application."""
    Config.create_directories()
    Config.display()

    random.seed(Config.RANDOM_SEED)
    torch.manual_seed(Config.RANDOM_SEED)

    # 1. Create the environment.
    environment = PathfindingEnvironment(
        **Config.get_environment_config()
    )

    # 2. The user creates obstacles by clicking grid cells.
    astar_path = AStar(environment.grid).find_path()

    # 3. Initialize the DQN trainer.
    trainer = Trainer(
        environment,
        **Config.get_training_config()
    )

    # 4. Create the visualization.
    renderer = Renderer(
        environment.grid,
        cell_size=Config.CELL_SIZE,
        sidebar_width=Config.SIDEBAR_WIDTH
    )
    renderer.fps = Config.FPS
    renderer.set_astar_path(astar_path)

    dashboard = Dashboard(
        renderer,
        trainer=trainer,
        environment=environment
    )

    # 5. Initialize statistics and charts.
    statistics = Statistics(
        output_file=Config.TRAINING_HISTORY_FILE
    )
    charts = TrainingCharts(Config.CHART_DIR)

    generation = 1
    agent_index = 0
    current_agent = None
    state = None
    episode_reward = 0.0
    episode_losses = []

    generation_rewards = []
    generation_successes = []
    generation_path_lengths = []

    training_finished = False

    print("Application started.")
    print("Use the dashboard buttons to control training.")
    print("Close the window to exit.")

    try:
        while dashboard.running:
            # Process window and button events.
            if not dashboard.handle_events():
                break

            controls = dashboard.get_controls()

            if (
                controls["training_started"]
                and statistics.training_start_time is None
            ):
                statistics.start_training()

            if controls["reset_requested"]:
                trainer.reset_training(reset_environment=False)
                statistics.reset()
                statistics.start_training()
                generation = 1
                agent_index = 0
                current_agent = None
                state = None
                episode_reward = 0.0
                episode_losses.clear()
                generation_rewards.clear()
                generation_successes.clear()
                generation_path_lengths.clear()
                training_finished = False
                renderer.set_agent_data()
                renderer.set_status("Training reset")
                dashboard.clear_requests()

            elif controls["next_environment_requested"]:
                trainer.reset_training(reset_environment=True)
                statistics.reset()
                statistics.start_training()
                astar_path = AStar(environment.grid).find_path()
                renderer.set_astar_path(astar_path)
                generation = 1
                agent_index = 0
                current_agent = None
                state = None
                episode_reward = 0.0
                episode_losses.clear()
                generation_rewards.clear()
                generation_successes.clear()
                generation_path_lengths.clear()
                training_finished = False
                renderer.set_agent_data()
                renderer.set_status("New environment")
                dashboard.clear_requests()

            # Recalculate the baseline after any wall edit.
            astar_path = AStar(environment.grid).find_path()
            renderer.set_astar_path(astar_path)

            # Start or resume an episode when training is enabled.
            if (
                controls["training_started"]
                and not controls["training_paused"]
                and not training_finished
            ):
                if current_agent is None:
                    if agent_index >= len(trainer.agents):
                        # All agents have completed this generation.
                        generation_record = statistics.record_generation(
                            generation=generation,
                            rewards=generation_rewards,
                            successes=generation_successes,
                            path_lengths=generation_path_lengths,
                            losses=episode_losses
                        )

                        print(
                            f"Generation {generation}: "
                            f"reward={generation_record['average_reward']}, "
                            f"success rate="
                            f"{generation_record['success_rate']}%"
                        )

                        # Update the target network periodically.
                        if (
                            generation
                            % Config.TARGET_UPDATE_FREQUENCY
                            == 0
                        ):
                            trainer.target_model.load_state_dict(
                                trainer.model.state_dict()
                            )

                        # Decay exploration after each generation.
                        for agent in trainer.agents:
                            agent.decay_epsilon()

                        generation += 1
                        agent_index = 0

                        generation_rewards.clear()
                        generation_successes.clear()
                        generation_path_lengths.clear()
                        episode_losses.clear()

                        if generation > Config.MAX_GENERATIONS:
                            training_finished = True
                            dashboard.training_started = False
                            renderer.set_status("Training completed")
                        else:
                            renderer.set_status(
                                f"Generation {generation}"
                            )

                    if not training_finished:
                        current_agent = trainer.agents[agent_index]

                        # Reset the environment for the next episode.
                        state = environment.reset()
                        current_agent.reset(environment.agent_position)

                        episode_reward = 0.0

                        renderer.set_agent_data(
                            position=environment.agent_position,
                            path=current_agent.path,
                            visited_positions=environment.visited_positions
                        )

                if current_agent is not None:
                    # Choose an action using the current DQN policy.
                    action = current_agent.select_action(
                        state,
                        training=True
                    )

                    next_state, reward, done, info = environment.step(
                        action
                    )

                    current_agent.remember(
                        state,
                        action,
                        reward,
                        next_state,
                        done
                    )

                    current_agent.update_after_step(
                        info["position"],
                        reward,
                        done,
                        success=info["success"]
                    )

                    state = next_state
                    episode_reward += reward

                    # Train the network when enough experiences exist.
                    loss = trainer.train_step()

                    if loss is not None:
                        episode_losses.append(float(loss))

                    # Update the live visualization.
                    renderer.set_agent_data(
                        position=environment.agent_position,
                        path=environment.path,
                        visited_positions=environment.visited_positions
                    )

                    renderer.set_status(
                        f"Generation {generation} | "
                        f"Agent {agent_index + 1}/{len(trainer.agents)}"
                    )

                    if done:
                        success = bool(info["success"])

                        generation_rewards.append(episode_reward)
                        generation_successes.append(success)

                        if success:
                            path_length = max(
                                0,
                                len(environment.path) - 1
                            )
                            generation_path_lengths.append(path_length)

                            statistics.record_path_comparison(
                                path_length,
                                max(0, len(astar_path) - 1)
                            )

                        statistics.record_episode(
                            agent_id=current_agent.agent_id,
                            reward=episode_reward,
                            steps=info["steps"],
                            success=success,
                            generation=generation
                        )

                        agent_index += 1
                        current_agent = None

            # Collect live summary metrics for the sidebar.
            summary = statistics.get_summary()

            live_statistics = {
                "Generation": min(
                    generation,
                    Config.MAX_GENERATIONS
                ),
                "Episodes": summary["total_episodes"],
                "Success rate": (
                    f"{summary['overall_success_rate']}%"
                ),
                "Avg reward": summary["overall_average_reward"],
                "Training time": (
                    f"{summary['training_duration_seconds']} s"
                )
            }

            # Draw the grid, statistics, and dashboard controls.
            dashboard.draw(live_statistics)
            renderer.tick()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        statistics.stop_training()

        # Save the training results.
        statistics.save_generation_history()
        statistics.save_episode_history(
            os.path.join(Config.DATA_DIR, "episode_history.csv")
        )
        statistics.print_summary()

        # Save the learned model.
        model_path = os.path.join(
            Config.CHECKPOINT_DIR,
            "dqn_model.pth"
        )

        torch.save(
            trainer.model.state_dict(),
            model_path
        )
        print(f"DQN model saved to: {model_path}")

        # Generate charts if any generations were completed.
        if statistics.generation_history:
            charts.plot_all(
                statistics.generation_history,
                save=True,
                show=False
            )

        renderer.close()


if __name__ == "__main__":
    main()