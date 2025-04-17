import os
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from data_processing import prepare_datasets  # Import your data preparation function
import torch
import argparse
from sklearn.metrics import f1_score, accuracy_score
from config import TRAINING_ARGS  # Import the dictionary


def compute_metrics(pred):
    """
    Computes evaluation metrics (F1 score and accuracy).
    Args:
        pred: A `Prediction` object from the `Trainer`.
    Returns:
        A dictionary containing the F1 score and accuracy.
    """
    labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    f1 = f1_score(labels, predictions)
    acc = accuracy_score(labels, predictions)
    return {"eval_f1": f1, "eval_accuracy": acc}


def train(model_name="roberta-base", output_dir="model/lora_test"):
    """
    Trains a sequence classification model using LoRA.
    Args:
        model_name (str): The name of the pre-trained model to use.
        output_dir (str): The directory where the trained model will be saved.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and verify data
    dataset = prepare_datasets()  # Ensure this returns a dict with 'train' and 'test'
    if not isinstance(dataset, dict) or "train" not in dataset or "test" not in dataset:
        raise ValueError(
            "prepare_datasets() must return a dictionary with 'train' and 'test' keys, containing Dataset objects."
        )

    print(f"\nFinal dataset sizes:")
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")

    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],  # Corrected target_modules
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Use the TRAINING_ARGS from config.py, and update the output_dir
    training_args = TRAINING_ARGS.copy()  # Create a copy to avoid modifying the original
    training_args["output_dir"] = output_dir  # Override with the command-line argument

    # Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,  # Pass the compute_metrics function
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
