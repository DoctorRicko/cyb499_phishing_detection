# /src/config.py
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# Model Settings
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Default model
TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"

# Training Params
TRAINING_ARGS = {
    "output_dir": str(MODEL_DIR / "fine_tuned"),
    "per_device_train_batch_size": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "evaluation_strategy": "epoch",
    "logging_dir": str(PROJECT_DIR / "logs")
}

# Data Processing
MAX_LENGTH = 256  # Token limit
TEST_SIZE = 0.2   # Validation split