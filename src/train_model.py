# /src/train_model.py
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from .config import TRAINING_ARGS, MODEL_NAME
from .data_processing import prepare_data
import torch

def train():
    # Load data
    data = prepare_data()
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Training setup
    training_args = TrainingArguments(**TRAINING_ARGS)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"]
    )
    
    # Execute training
    trainer.train()
    trainer.save_model()
    
if __name__ == "__main__":
    train()