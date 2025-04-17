from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from .config import TRAINING_ARGS # Import the dictionary, but we'll modify it
from .data_processing import prepare_datasets
import torch
import argparse
import os

def setup_lora(model):
    """Configure LoRA adapters for RoBERTa"""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Target the query and value projection layers in the self-attention modules
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    return get_peft_model(model, config)

def train():
    parser = argparse.ArgumentParser(description="Train a LoRA model for phishing detection")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Pre-trained model name")
    parser.add_argument("--dataset_path", type=str, default="data/processed", help="Path to the processed dataset")
    parser.add_argument("--output_dir", type=str, default="model/lora_v1", help="Output directory for the trained model")
    args = parser.parse_args()

    print(f"Using model: {args.model_name}")

    # Load data - pass the model_name argument to prepare_datasets
    tokenized = prepare_datasets(args.model_name)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Apply LoRA
    model = setup_lora(model)
    model.print_trainable_parameters()

    # Create a copy of TRAINING_ARGS to avoid modifying the original config
    training_args = TRAINING_ARGS.copy()
    training_args["output_dir"] = args.output_dir # Override output_dir

    # Print the training arguments
    print("Training Arguments:")
    for key, value in training_args.items():
        print(f"  {key}: {value} ({type(value)})")  # Print key, value, and type

    # Train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args), # Use the modified dictionary
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"]
    )
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    train()
