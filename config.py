"""
Configuration for Chess AI
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Neural Network Configuration
INPUT_CHANNELS = 20
RESIDUAL_BLOCKS = 10
FILTERS = 256
HIDDEN_UNITS = 256
POLICY_OUTPUT_SIZE = 4672

# MCTS Configuration
MCTS_SIMULATIONS = 200
MCTS_C_PUCT = 1.5
MCTS_TEMPERATURE = 1.0

# Training Configuration
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 10
SELF_PLAY_GAMES = 100
TRAINING_ITERATIONS = 10

# Imitation Learning Configuration
IL_EPOCHS = 5
IL_BATCH_SIZE = 64
IL_LEARNING_RATE = 0.001
IL_MAX_GAMES = 1000

# Paths
MODEL_PATH = "models/model.pth"
DATA_PATH = "data/"
LOG_PATH = "logs/"

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '3306'))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '123369')
DB_NAME = os.getenv('DB_NAME', 'chess_ai')
