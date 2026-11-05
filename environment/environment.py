# environment/environment.py

from .grid import Grid
from .maze_generator import MazeGenerator


class PathfindingEnvironment:
    # Available actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1)
    }

    def __init__(self, rows=20, cols=20, max_steps=400):
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps

        self.grid = Grid(rows, cols)
        self.maze_generator = MazeGenerator(self.grid)

        self.agent_position = None
        self.steps = 0
        self.done = False
        self.success = False

        self.visited_positions = set()
        self.path = []

        # Generate the initial start and goal.
        self.reset()

    def reset(self):
        """
        Reset the current episode.

        The grid, walls, start, and goal remain unchanged.
        Only the agent's episode state is reset.
        """
        if self.grid.start is None or self.grid.goal is None:
            self.maze_generator.generate_positions()

        self.agent_position = self.grid.start

        self.steps = 0
        self.done = False
        self.success = False

        self.visited_positions = {
            self.agent_position
        }

        self.path = [self.agent_position]

        return self.get_state()

    def reset_environment(self):
        """
        Create a completely new environment.

        This clears the walls and generates new start
        and goal positions.
        """
        self.maze_generator.generate_new_environment()

        return self.reset()

    def get_state(self):
        """
        Return an eight-element numerical state.

        State representation:
        1. Agent row
        2. Agent column
        3. Goal row
        4. Goal column
        5. Obstacle above
        6. Obstacle below
        7. Obstacle to the left
        8. Obstacle to the right

        Coordinates are normalized between 0 and 1.
        Obstacle indicators are 1 if blocked and 0 otherwise.
        """
        row, col = self.agent_position
        goal_row, goal_col = self.grid.goal

        max_row = max(1, self.rows - 1)
        max_col = max(1, self.cols - 1)

        state = [
            row / max_row,
            col / max_col,
            goal_row / max_row,
            goal_col / max_col
        ]

        # Check obstacles in four directions.
        for action in range(4):
            dr, dc = self.ACTIONS[action]

            neighbor = (
                row + dr,
                col + dc
            )

            blocked = not self.grid.is_walkable(neighbor)

            state.append(1.0 if blocked else 0.0)

        return state

    def get_manhattan_distance(self, position=None):
        """
        Calculate Manhattan distance from a position
        to the goal.
        """
        if position is None:
            position = self.agent_position

        row, col = position
        goal_row, goal_col = self.grid.goal

        return (
            abs(row - goal_row)
            + abs(col - goal_col)
        )

    def step(self, action):
        """
        Perform one action and return:

        next_state, reward, done, info

        The four-element return format is intentionally
        compatible with a simple DQN training loop.
        """
        if self.done:
            return (
                self.get_state(),
                0.0,
                True,
                self._get_info()
            )

        if action not in self.ACTIONS:
            raise ValueError(
                "Invalid action. Use 0=UP, 1=DOWN, "
                "2=LEFT, or 3=RIGHT."
            )

        old_position = self.agent_position
        old_distance = self.get_manhattan_distance()

        dr, dc = self.ACTIONS[action]

        new_position = (
            old_position[0] + dr,
            old_position[1] + dc
        )

        self.steps += 1

        # Case 1: The agent tries to enter a wall
        # or move outside the grid.
        if not self.grid.is_walkable(new_position):
            reward = -20.0
            self.done = True
            self.success = False

            return (
                self.get_state(),
                reward,
                self.done,
                self._get_info()
            )

        # Update the agent position.
        self.agent_position = new_position
        self.path.append(new_position)

        # Case 2: The agent reaches the goal.
        if new_position == self.grid.goal:
            reward = 100.0

            self.done = True
            self.success = True

            return (
                self.get_state(),
                reward,
                self.done,
                self._get_info()
            )

        # Case 3: Calculate the reward for movement. In addition to
        # rewarding progress, provide dense feedback based on proximity
        # so the agent can distinguish states that are closer to the goal.
        new_distance = self.get_manhattan_distance()
        maximum_distance = max(1, self.rows + self.cols - 2)
        proximity_reward = (
            2.0 * (1.0 - new_distance / maximum_distance)
        )

        if new_distance < old_distance:
            reward = 2.0 + proximity_reward
        elif new_distance > old_distance:
            reward = -2.0 + proximity_reward
        else:
            reward = -0.5 + proximity_reward

        # Penalize revisiting previously visited positions.
        if new_position in self.visited_positions:
            reward -= 5.0

        self.visited_positions.add(new_position)

        # Case 4: The maximum step limit is reached.
        if self.steps >= self.max_steps:
            self.done = True
            self.success = False

        return (
            self.get_state(),
            reward,
            self.done,
            self._get_info()
        )

    def _get_info(self):
        """Return useful episode information."""
        return {
            "position": self.agent_position,
            "goal": self.grid.goal,
            "steps": self.steps,
            "success": self.success,
            "done": self.done,
            "distance_to_goal": self.get_manhattan_distance(),
            "path_length": max(0, len(self.path) - 1)
        }

    def get_valid_actions(self):
        """
        Return actions that lead to walkable cells.

        Useful for debugging and visualization.
        """
        valid_actions = []

        row, col = self.agent_position

        for action, (dr, dc) in self.ACTIONS.items():
            next_position = (
                row + dr,
                col + dc
            )

            if self.grid.is_walkable(next_position):
                valid_actions.append(action)

        return valid_actions

    def render_text(self):
        """
        Print a text-based representation of the grid.

        Useful for testing before implementing Pygame.
        """
        symbols = {
            Grid.EMPTY: ".",
            Grid.WALL: "#",
            Grid.START: "S",
            Grid.GOAL: "G"
        }

        display = [
            [symbols[cell] for cell in row]
            for row in self.grid.cells
        ]

        if self.agent_position is not None:
            row, col = self.agent_position

            if self.agent_position != self.grid.goal:
                display[row][col] = "A"

        print(
            "\n".join(
                " ".join(row)
                for row in display
            )
        )