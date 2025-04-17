from data_processing import prepare_datasets
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
import torch
import os

def prepare_datasets():
    """Returns tokenized datasets with train/test split"""
    # Load and balance data
    legit_df = load_enron_emails()
    phish_df = load_phishing_data()
    
    # Balance classes
    min_samples = min(len(legit_df), len(phish_df))
    combined = pd.concat([
        legit_df.sample(min_samples),
        phish_df.sample(min_samples)
    ])
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dataset = Dataset.from_pandas(combined)
    
    return dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        ),
        batched=True
    ).train_test_split(test_size=0.2)

def train():
    # Configuration
    model_name = "roberta-base"
    output_dir = "model/lora_roberta_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    dataset = prepare_datasets()
    print(f"\nDataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],  # Different for RoBERTa
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLASSIFICATION"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )
    
    # Train
    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    train()