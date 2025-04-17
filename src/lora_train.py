from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from .config import TRAINING_ARGS, MODEL_NAME
from .data_processing import load_and_prepare_data, tokenize_dataset
import torch

def setup_lora(model):
    """Configure LoRA adapters"""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLASSIFICATION"
    )
    return get_peft_model(model, config)

def train():
    # Load data
    df = load_and_prepare_data()
    tokenized = tokenize_dataset(df)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Apply LoRA
    model = setup_lora(model)
    model.print_trainable_parameters()
    
    # Train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**TRAINING_ARGS),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"]
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    train()