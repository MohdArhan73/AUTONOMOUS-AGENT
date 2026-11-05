# environment/maze_generator.py

import random

from .grid import Grid


class MazeGenerator:
    def __init__(self, grid):
        self.grid = grid

    def generate_positions(self):
        """
        Generate random start and goal positions.

        Both positions must:
        - Be inside the grid.
        - Be different from each other.
        - Not overlap any wall.
        """
        available_positions = self.grid.get_empty_positions()

        if len(available_positions) < 2:
            raise ValueError(
                "At least two empty cells are required "
                "to generate start and goal positions."
            )

        start, goal = random.sample(available_positions, 2)

        self.grid.set_start(start)
        self.grid.set_goal(goal)

        return start, goal

    def generate_new_environment(self):
        """
        Create a fresh environment with no walls
        and randomly generated start and goal positions.
        """
        self.grid.reset()

        return self.generate_positions()

    def generate_random_walls(self, wall_count):
        """
        Optionally generate random walls.

        This method is provided for future testing.
        It is not called automatically, because the
        user should normally design the maze manually.
        """
        available_positions = [
            position
            for position in self.grid.get_empty_positions()
            if position != self.grid.start
            and position != self.grid.goal
        ]

        wall_count = max(0, min(wall_count, len(available_positions)))

        selected_positions = random.sample(
            available_positions,
            wall_count
        )

        for position in selected_positions:
            self.grid.add_wall(position)

        return selected_positions

    def generate_random_scenario(self, wall_count=0):
        """
        Generate a new scenario with random start,
        goal, and an optional number of random walls.
        """
        start, goal = self.generate_new_environment()

        if wall_count > 0:
            self.generate_random_walls(wall_count)

        return {
            "start": start,
            "goal": goal,
            "walls": set(self.grid.walls)
        }