# algorithms/astar.py

import heapq
import time


class AStar:
    """A* shortest-path algorithm for grid-based environments."""

    def __init__(self, grid):
        self.grid = grid

        self.path = []
        self.path_length = None
        self.execution_time = 0.0
        self.nodes_explored = 0

    @staticmethod
    def heuristic(position, goal):
        """Calculate Manhattan distance between two positions."""
        return (
            abs(position[0] - goal[0])
            + abs(position[1] - goal[1])
        )

    def find_path(self, start=None, goal=None):
        """
        Find the shortest path from start to goal.

        Args:
            start: Starting position as (row, column).
            goal: Destination as (row, column).

        Returns:
            A list of positions from start to goal,
            or an empty list if no valid path exists.
        """
        start_time = time.perf_counter()

        self.path = []
        self.path_length = None
        self.nodes_explored = 0

        if start is None:
            start = self.grid.start

        if goal is None:
            goal = self.grid.goal

        # Validate start and goal.
        if (
            start is None
            or goal is None
            or not self.grid.is_valid_position(start)
            or not self.grid.is_valid_position(goal)
            or self.grid.is_wall(start)
            or self.grid.is_wall(goal)
        ):
            self.execution_time = time.perf_counter() - start_time
            return []

        # A path from a position to itself has zero length.
        if start == goal:
            self.path = [start]
            self.path_length = 0
            self.execution_time = time.perf_counter() - start_time
            return self.path

        # Priority queue entries:
        # (estimated_total_cost, tie_breaker, position)
        open_set = []
        counter = 0

        heapq.heappush(
            open_set,
            (self.heuristic(start, goal), counter, start)
        )

        # Cost of reaching each position.
        g_score = {start: 0}

        # Used to reconstruct the final path.
        came_from = {}

        # Prevent unnecessary reprocessing.
        closed_set = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            self.nodes_explored += 1

            # The destination has been reached.
            if current == goal:
                self.path = self._reconstruct_path(
                    came_from,
                    current
                )

                self.path_length = len(self.path) - 1
                break

            # Explore the four neighboring cells.
            for neighbor in self.grid.get_neighbors(current):

                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score.get(
                    neighbor, float("inf")
                ):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score

                    f_score = (
                        tentative_g_score
                        + self.heuristic(neighbor, goal)
                    )

                    counter += 1

                    heapq.heappush(
                        open_set,
                        (f_score, counter, neighbor)
                    )

        self.execution_time = time.perf_counter() - start_time

        return self.path

    @staticmethod
    def _reconstruct_path(came_from, current):
        """Reconstruct the path from start to goal."""
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return path

    def get_results(self):
        """Return pathfinding results for the dashboard."""
        return {
            "path": self.path,
            "path_found": bool(self.path),
            "path_length": self.path_length,
            "execution_time": self.execution_time,
            "nodes_explored": self.nodes_explored
        }