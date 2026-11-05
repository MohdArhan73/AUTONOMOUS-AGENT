import os


class Config:
    # Project paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    CHART_DIR = os.path.join(DATA_DIR, "charts")

    # Grid environment
    GRID_ROWS = 20
    GRID_COLS = 20
    MAX_STEPS = 400
    WALL_COUNT = 60

    # Multi-agent training
    NUM_AGENTS = 20
    MAX_GENERATIONS = 500
    TRAINING_UPDATES = 100
    BATCH_SIZE = 64
    REPLAY_CAPACITY = 50000

    # DQN model
    STATE_SIZE = 8
    ACTION_SIZE = 4
    HIDDEN_SIZE = 128
    LEARNING_RATE = 0.001
    GAMMA = 0.95

    # Exploration settings
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995

    # Target network
    TARGET_UPDATE_FREQUENCY = 10

    # Visualization
    CELL_SIZE = 30
    SIDEBAR_WIDTH = 280
    FPS = 30
    SHOW_ASTAR_PATH = True

    # Reproducibility
    RANDOM_SEED = 42

    # Output files
    TRAINING_HISTORY_FILE = os.path.join(
        DATA_DIR,
        "training_history.csv"
    )

    @classmethod
    def create_directories(cls):
        """Create folders needed to save project outputs."""
        directories = [
            cls.DATA_DIR,
            cls.MODEL_DIR,
            cls.CHECKPOINT_DIR,
            cls.CHART_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_training_config(cls):
        """Return the settings required by the Trainer."""
        return {
            "num_agents": cls.NUM_AGENTS,
            "max_generations": cls.MAX_GENERATIONS,
            "batch_size": cls.BATCH_SIZE,
            "learning_rate": cls.LEARNING_RATE,
            "gamma": cls.GAMMA,
            "epsilon": cls.EPSILON_START,
            "epsilon_min": cls.EPSILON_MIN,
            "epsilon_decay": cls.EPSILON_DECAY,
            "replay_capacity": cls.REPLAY_CAPACITY,
            "target_update_frequency": cls.TARGET_UPDATE_FREQUENCY,
            "training_updates": cls.TRAINING_UPDATES,
            "seed": cls.RANDOM_SEED
        }

    @classmethod
    def get_environment_config(cls):
        """Return the settings required by the environment."""
        return {
            "rows": cls.GRID_ROWS,
            "cols": cls.GRID_COLS,
            "max_steps": cls.MAX_STEPS
        }

    @classmethod
    def display(cls):
        """Print the main project configuration."""
        print("\n========== PROJECT CONFIGURATION ==========")
        print(f"Grid size: {cls.GRID_ROWS} x {cls.GRID_COLS}")
        print(f"Number of agents: {cls.NUM_AGENTS}")
        print(f"Maximum generations: {cls.MAX_GENERATIONS}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Discount factor: {cls.GAMMA}")
        print(f"Initial epsilon: {cls.EPSILON_START}")
        print(f"Minimum epsilon: {cls.EPSILON_MIN}")
        print("===========================================\n")