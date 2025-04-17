# src/lora_train.py
from transformers import TrainingArguments, Trainer
from .config import TRAINING_ARGS
from .data_processing import prepare_data
from .lora_utils import prepare_lora_model
import torch

def train_lora():
    # Initialize
    model = prepare_lora_model(MODEL_NAME)
    data = prepare_data()
    
    # Training arguments (override some defaults)
    args = TrainingArguments(
        **TRAINING_ARGS,
        report_to="none",          # Change to "wandb" if using
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        compute_metrics=compute_metrics  # From evaluate.py
    )
    
    # Start training
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    trainer.train()
    
    # Save only adapters (small files)
    model.save_pretrained(str(MODEL_DIR / "lora_adapters"))

if __name__ == "__main__":
    train_lora()