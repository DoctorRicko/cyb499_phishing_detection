import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

from data_processing import prepare_datasets
from config import DATA_DIR, MAX_LENGTH
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
import os

def train(model_name="roberta-base", output_dir="model/lora_roberta_v1"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    dataset = prepare_datasets()
    print(f"Loaded {len(dataset['train'])} training samples")
    
    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value", "key"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLASSIFICATION"
    )
    model = get_peft_model(model, peft_config)
    
    # Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", default="model/lora_roberta_v1")
    args = parser.parse_args()
    
    train(args.model_name, args.output_dir)