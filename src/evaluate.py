import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,  # Import f1_score
    accuracy_score
)
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer  # Import AutoModelForSequenceClassification and AutoTokenizer
from pathlib import Path
#from .config import DATA_DIR  # Remove this line
import matplotlib.pyplot as plt
import torch
import argparse # Import argparse


def evaluate(model_path, test_file, results_dir):  # Add arguments
    """
    Evaluates a fine-tuned model for text classification.

    Args:
        model_path (str): Path to the directory containing the trained model.
        test_file (str): Path to the test data CSV file.
        results_dir (str): Path to the directory to save the evaluation results.
    """
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,  # Assuming binary classification
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu") # Move model to device

    # Load test data
    test_data = pd.read_csv(test_file)
    if "text" not in test_data.columns or "label" not in test_data.columns:
        raise ValueError("Test data must contain 'text' and 'label' columns")

    # Tokenize the test data
    inputs = tokenizer(
        test_data["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=256,  # Or your desired max length
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")  # Move inputs to device

    # Run predictions
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    y_pred = predictions
    y_true = test_data["label"].values

    # Generate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(Path(results_dir) / "metrics.json", "w") as f:
        json.dump(report, f)
    print(f"Metrics saved to {Path(results_dir) / 'metrics.json'}")

    # Calculate and save F1 score and accuracy
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    with open(Path(results_dir) / "f1_and_accuracy.json", "w") as f:
        json.dump({"f1": f1, "accuracy": acc}, f)
    print(f"F1 and Accuracy saved to {Path(results_dir) / 'f1_and_accuracy.json'}")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Phishing"])
    disp.plot()
    plt.savefig(Path(results_dir) / "confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to {Path(results_dir) / 'confusion_matrix.png'}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a text classification model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/processed/test.csv",
        help="Path to the test data CSV file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to the directory to save the evaluation results",
    )
    args = parser.parse_args()

    evaluate(args.model_path, args.test_file, args.results_dir)
