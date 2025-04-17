import os
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from data_processing import prepare_datasets
import torch

def train(model_name="roberta-base", output_dir="model/lora_test"):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and verify data
    dataset = prepare_datasets()
    print(f"\nFinal dataset sizes:")
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value", "key"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="none",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"\nTraining complete! Model saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="roberta-base")
    parser.add_argument("--output_dir", default="model/lora_test")
    args = parser.parse_args()
    
    train(args.model_name, args.output_dir)