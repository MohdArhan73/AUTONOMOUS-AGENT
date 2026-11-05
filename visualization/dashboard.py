import pygame


class Dashboard:
    def __init__(self, renderer, trainer=None, environment=None):
        self.renderer = renderer
        self.trainer = trainer
        self.environment = environment

        self.running = True
        self.training_started = False
        self.training_paused = False
        self.show_astar = True
        self.reset_requested = False
        self.next_environment_requested = False

        self.buttons = {}
        self.hovered_button = None

        self.button_width = 118
        self.button_height = 32
        self.button_gap = 8

        self._create_buttons()

    def _create_buttons(self):
        """Create a compact two-column control panel."""
        sidebar_left = self.renderer.grid_width + 15
        panel_bottom = self.renderer.screen_height - 14
        row_height = self.button_height + self.button_gap
        labels = [
            ("Start", 0, 0),
            ("Pause", 1, 0),
            ("Resume", 0, 1),
            ("Reset", 1, 1),
            ("Next Environment", 0, 2),
            ("Toggle A*", 1, 2),
        ]

        for name, column, row in labels:
            x = sidebar_left + column * (self.button_width + self.button_gap)
            y = panel_bottom - self.button_height - row * row_height
            self.buttons[name] = pygame.Rect(
                x, y, self.button_width, self.button_height
            )

    def draw_buttons(self):
        """Draw all dashboard buttons."""
        screen = self.renderer.screen
        font = self.renderer.small_font

        for name, rect in self.buttons.items():
            is_active = (
                name == "Start" and self.training_started
            ) or (
                name == "Pause" and self.training_paused
            ) or (
                name == "Toggle A*" and self.show_astar
            )
            color = (45, 122, 184) if is_active else (55, 75, 96)
            if name == self.hovered_button:
                color = (70, 145, 205)

            pygame.draw.rect(
                screen,
                color,
                rect,
                border_radius=5
            )

            pygame.draw.rect(
                screen,
                (120, 145, 165),
                rect,
                1,
                border_radius=5
            )

            label_font = font
            if font.size(name)[0] > rect.width - 12:
                label_font = pygame.font.SysFont("Arial", 12)
            label = label_font.render(name, True, (245, 248, 250))
            label_rect = label.get_rect(center=rect.center)
            screen.blit(label, label_rect)

    def handle_button(self, name):
        """Execute the selected dashboard action."""
        if name == "Start":
            self.training_started = True
            self.training_paused = False

        elif name == "Pause":
            if self.training_started:
                self.training_paused = True

        elif name == "Resume":
            if self.training_started:
                self.training_paused = False

        elif name == "Reset":
            self.training_started = False
            self.training_paused = False
            self.reset_requested = True

        elif name == "Next Environment":
            self.training_started = False
            self.training_paused = False
            self.next_environment_requested = True

        elif name == "Toggle A*":
            self.show_astar = not self.show_astar
            self.renderer.show_astar_path = self.show_astar

    def handle_events(self):
        """
        Handle window and button events.

        Returns False when the user closes the window.
        """
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False
                return False

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                for name, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        self.handle_button(name)
                        break
                else:
                    self._edit_wall(event.pos)

            if event.type == pygame.MOUSEMOTION:
                self.hovered_button = None
                for name, rect in self.buttons.items():
                    if rect.collidepoint(event.pos):
                        self.hovered_button = name
                        break

        return self.running

    def _edit_wall(self, mouse_position):
        """Toggle a wall while the editor is not actively training."""
        if self.training_started and not self.training_paused:
            return

        row = mouse_position[1] // self.renderer.cell_size
        col = mouse_position[0] // self.renderer.cell_size
        position = (row, col)

        if self.environment.grid.is_valid_position(position):
            self.environment.grid.toggle_wall(position)
            self.renderer.set_astar_path(
                self.renderer.astar_path
            )

    def get_controls(self):
        """Return the current dashboard control states."""
        return {
            "running": self.running,
            "training_started": self.training_started,
            "training_paused": self.training_paused,
            "show_astar": self.show_astar,
            "reset_requested": self.reset_requested,
            "next_environment_requested": (
                self.next_environment_requested
            )
        }

    def clear_requests(self):
        """Clear one-shot reset actions after the main loop handles them."""
        self.reset_requested = False
        self.next_environment_requested = False

    def draw(self, statistics=None):
        """Draw the renderer and dashboard controls."""
        self.renderer.draw(statistics)
        self.draw_buttons()

        pygame.display.flip()