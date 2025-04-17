from pathlib import Path

# Path Configuration (Verified ✅)
PROJECT_DIR = Path(__file__).parent.parent  # Correct project root
DATA_DIR = PROJECT_DIR / "data"  # Confirmed structure matches your project
MODEL_DIR = PROJECT_DIR / "models"

# Data Paths (Add validation)
RAW_DIR = DATA_DIR / "raw"  # For original datasets
PROCESSED_DIR = DATA_DIR / "processed"  # For cleaned data
EXTRACTED_DIR = DATA_DIR / "extracted"  # For unpacked files

# Model Configuration (Updated for RoBERTa)
MODEL_NAME = "roberta-base"  # Changed from LLaMA-2 (better for your task)
TOKENIZER_NAME = MODEL_NAME

# Training Parameters (Optimized)
TRAINING_ARGS = {
    "output_dir": str(MODEL_DIR / "lora_adapters"),
    "per_device_train_batch_size": 2,  # Reduced for GPU safety
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "fp16": True,  # Enable mixed-precision
    "logging_dir": str(PROJECT_DIR / "logs"),
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,  # Added
    "metric_for_best_model": "f1"  # Added
}

# Data Processing (Enhanced)
MAX_LENGTH = 384  # Increased from 256 (matches your earlier analysis)
TEST_SIZE = 0.2
KEYWORDS = ['click', 'verify', 'account', *EXTRA_KEYWORDS]  # Combined list

# Enron Config (Optional - Only if using maildir)
ENRON_CONFIG = {
    "max_emails": 10_000,
    "allowed_folders": ["inbox", "sent_items"]  # Fixed common folder name
}