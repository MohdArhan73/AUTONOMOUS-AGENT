# environment/grid.py

class Grid:
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3

    def __init__(self, rows=20, cols=20):
        self.rows = rows
        self.cols = cols

        # Every cell initially contains an empty space.
        self.cells = [
            [self.EMPTY for _ in range(cols)]
            for _ in range(rows)
        ]

        self.start = None
        self.goal = None
        self.walls = set()

    def is_valid_position(self, position):
        """Check whether a position lies inside the grid."""
        if position is None:
            return False

        row, col = position

        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
        )

    def is_wall(self, position):
        """Return True if the given position contains a wall."""
        return position in self.walls

    def is_walkable(self, position):
        """Check whether an agent can enter a cell."""
        return (
            self.is_valid_position(position)
            and position not in self.walls
        )

    def set_start(self, position):
        """Set the starting position."""
        if not self.is_valid_position(position):
            raise ValueError("Start position is outside the grid.")

        if position in self.walls:
            raise ValueError("Start position cannot be a wall.")

        if position == self.goal:
            raise ValueError("Start and goal cannot overlap.")

        self.start = position
        self._update_cells()

    def set_goal(self, position):
        """Set the destination position."""
        if not self.is_valid_position(position):
            raise ValueError("Goal position is outside the grid.")

        if position in self.walls:
            raise ValueError("Goal position cannot be a wall.")

        if position == self.start:
            raise ValueError("Start and goal cannot overlap.")

        self.goal = position
        self._update_cells()

    def add_wall(self, position):
        """Add a wall to an empty cell."""
        if not self.is_valid_position(position):
            return False

        if position == self.start or position == self.goal:
            return False

        self.walls.add(position)
        self._update_cells()

        return True

    def remove_wall(self, position):
        """Remove a wall from the specified cell."""
        if position not in self.walls:
            return False

        self.walls.remove(position)
        self._update_cells()

        return True

    def toggle_wall(self, position):
        """Add a wall or remove it if it already exists."""
        if position in self.walls:
            return self.remove_wall(position)

        return self.add_wall(position)

    def clear_walls(self):
        """Remove every wall from the grid."""
        self.walls.clear()
        self._update_cells()

    def reset(self):
        """Reset the complete grid."""
        self.walls.clear()
        self.start = None
        self.goal = None

        self.cells = [
            [self.EMPTY for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    def _update_cells(self):
        """Update the cell matrix after grid modifications."""
        self.cells = [
            [self.EMPTY for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

        for row, col in self.walls:
            self.cells[row][col] = self.WALL

        if self.start is not None:
            row, col = self.start
            self.cells[row][col] = self.START

        if self.goal is not None:
            row, col = self.goal
            self.cells[row][col] = self.GOAL

    def get_neighbors(self, position):
        """Return valid, walkable neighboring positions."""
        row, col = position

        candidates = [
            (row - 1, col),  # Up
            (row + 1, col),  # Down
            (row, col - 1),  # Left
            (row, col + 1)   # Right
        ]

        return [
            neighbor
            for neighbor in candidates
            if self.is_walkable(neighbor)
        ]

    def get_all_positions(self):
        """Return every position in the grid."""
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
        ]

    def get_empty_positions(self):
        """Return all cells that do not contain walls."""
        return [
            position
            for position in self.get_all_positions()
            if position not in self.walls
        ]

    def __str__(self):
        """Return a simple text representation of the grid."""
        symbols = {
            self.EMPTY: ".",
            self.WALL: "#",
            self.START: "S",
            self.GOAL: "G"
        }

        return "\n".join(
            " ".join(symbols[cell] for cell in row)
            for row in self.cells
        )