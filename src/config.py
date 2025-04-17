from pathlib import Path

# Paths
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED_DIR = DATA_DIR / "extracted"

ENRON_CONFIG = {
    "max_emails": 10000,  # Limit if needed
    "allowed_folders": ["inbox", "sent"]  # Skip drafts, etc.
}

# Model
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
TOKENIZER_NAME = MODEL_NAME

# Training
TRAINING_ARGS = {
    "output_dir": str(MODEL_DIR / "lora_adapters"),
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "fp16": True,
    "logging_dir": str(PROJECT_DIR / "logs"),
    "evaluation_strategy": "epoch"
}

# Data
MAX_LENGTH = 256
TEST_SIZE = 0.2