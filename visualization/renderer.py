import pygame


class Renderer:
    # Colors
    WHITE = (245, 247, 250)
    BLACK = (25, 31, 38)
    GRAY = (190, 198, 208)
    DARK_GRAY = (80, 92, 105)
    GREEN = (40, 180, 90)
    RED = (220, 60, 60)
    BLUE = (50, 120, 220)
    YELLOW = (245, 200, 50)
    PURPLE = (150, 80, 200)
    CYAN = (40, 190, 200)

    def __init__(
        self,
        grid,
        cell_size=30,
        sidebar_width=280,
        title="Multi-Agent Pathfinding"
    ):
        pygame.init()

        self.grid = grid
        self.cell_size = cell_size
        self.sidebar_width = sidebar_width
        self.title = title

        self.grid_width = grid.cols * cell_size
        self.grid_height = grid.rows * cell_size

        self.screen_width = self.grid_width + sidebar_width
        self.screen_height = self.grid_height

        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height)
        )

        pygame.display.set_caption(title)

        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 14)
        self.large_font = pygame.font.SysFont("Arial", 24, bold=True)

        self.clock = pygame.time.Clock()
        self.fps = 30

        self.agent_position = None
        self.agent_path = []
        self.visited_positions = set()
        self.astar_path = []
        self.show_astar_path = True

        self.status_message = "Ready"

    def set_agent_data(
        self,
        position=None,
        path=None,
        visited_positions=None
    ):
        """Update the agent information shown on the grid."""
        self.agent_position = position
        self.agent_path = list(path) if path else []

        self.visited_positions = (
            set(visited_positions) if visited_positions else set()
        )

    def set_astar_path(self, path):
        """Set the path found by A* for visual comparison."""
        self.astar_path = list(path) if path else []

    def set_status(self, message):
        self.status_message = str(message)

    def draw_grid(self):
        """Draw the grid cells and their contents."""
        grid_background = (225, 230, 236)
        pygame.draw.rect(
            self.screen,
            grid_background,
            (0, 0, self.grid_width, self.grid_height)
        )

        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                position = (row, col)

                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                color = self.WHITE

                if position in self.grid.walls:
                    color = self.BLACK

                elif position in self.visited_positions:
                    color = (210, 230, 255)

                if self.show_astar_path and position in self.astar_path:
                    if position not in self.grid.walls:
                        color = (255, 235, 170)

                if position in self.agent_path:
                    if position not in self.grid.walls:
                        color = (170, 210, 255)

                if position == self.grid.start:
                    color = self.GREEN

                if position == self.grid.goal:
                    color = self.RED

                if position == self.agent_position:
                    color = self.BLUE

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(
                    self.screen,
                    (205, 212, 220),
                    rect,
                    1
                )

                # Draw labels on start and goal cells.
                if position == self.grid.start:
                    self._draw_cell_label("S", rect)

                elif position == self.grid.goal:
                    self._draw_cell_label("G", rect)

                elif position == self.agent_position:
                    self._draw_cell_label("A", rect)

    def _draw_cell_label(self, label, rect):
        text = self.font.render(label, True, self.WHITE)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def draw_sidebar(self, statistics=None):
        """Draw the information panel on the right side."""
        sidebar_x = self.grid_width

        pygame.draw.rect(
            self.screen,
            (30, 42, 55),
            (sidebar_x, 0, self.sidebar_width, self.screen_height)
        )

        x = sidebar_x + 15
        y = 15

        title = self.large_font.render(
            "PATHFINDING LAB",
            True,
            (245, 248, 250)
        )
        self.screen.blit(title, (x, y))
        subtitle = self.small_font.render(
            "Multi-agent reinforcement learning",
            True,
            (163, 183, 199)
        )
        self.screen.blit(subtitle, (x, y + 31))

        y += 62
        y = self._draw_panel(
            "RUN STATUS",
            [
                ("State", self.status_message),
                ("Agent", self.agent_position or "Waiting"),
            ],
            y,
            accent=(80, 190, 155)
        )
        y += 10
        y = self._draw_panel(
            "ENVIRONMENT",
            [
                ("Grid", f"{self.grid.rows} x {self.grid.cols}"),
                ("Start", str(self.grid.start)),
                ("Goal", str(self.grid.goal)),
            ],
            y,
            accent=(85, 155, 220)
        )
        y += 10
        if statistics:
            self._draw_panel(
                "TRAINING METRICS",
                list(statistics.items()),
                y,
                accent=(235, 180, 85)
            )

        self._draw_legend()

    def _draw_panel(self, heading, rows, y, accent):
        """Draw a dark information card and return the next available y."""
        panel_x = self.grid_width + 12
        panel_width = self.sidebar_width - 24
        row_height = 18
        panel_height = 26 + len(rows) * row_height
        pygame.draw.rect(
            self.screen,
            (39, 54, 68),
            (panel_x, y, panel_width, panel_height),
            border_radius=7
        )
        pygame.draw.rect(
            self.screen,
            accent,
            (panel_x, y, 4, panel_height),
            border_radius=2
        )
        heading_surface = self.small_font.render(
            heading,
            True,
            (165, 185, 201)
        )
        self.screen.blit(heading_surface,         (panel_x + 13, y + 6))

        for index, (label, value) in enumerate(rows):
            row_y = y + 25 + index * row_height
            label_surface = self.small_font.render(
                label,
                True,
                (157, 177, 193)
            )
            value_surface = self.small_font.render(
                str(value)[:22],
                True,
                (242, 246, 249)
            )
            self.screen.blit(label_surface, (panel_x + 13, row_y))
            self.screen.blit(
                value_surface,
                (panel_x + panel_width - value_surface.get_width() - 12, row_y)
            )

        return y + panel_height

    def _draw_legend(self):
        """Draw a compact legend above the controls."""
        x = self.grid_width + 15
        y = self.screen_height - 195
        legend = [
            ("S", self.GREEN, "Start"),
            ("G", self.RED, "Goal"),
            ("A", self.BLUE, "Agent"),
            ("", (255, 235, 170), "A* path"),
        ]
        label = self.small_font.render("LEGEND", True, (165, 185, 201))
        self.screen.blit(label, (x, y))
        for index, (marker, color, name) in enumerate(legend):
            item_x = x + (index % 2) * 130
            item_y = y + 20 + (index // 2) * 19
            pygame.draw.rect(
                self.screen,
                color,
                (item_x, item_y, 13, 13),
                border_radius=3
            )
            text = self.small_font.render(name, True, (225, 233, 239))
            self.screen.blit(text, (item_x + 19, item_y - 1))

    def draw(self, statistics=None):
        """Redraw the entire window."""
        self.screen.fill(self.WHITE)

        self.draw_grid()
        self.draw_sidebar(statistics)

    def handle_events(self):
        """Process window events. Return False when closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        return True

    def tick(self):
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()