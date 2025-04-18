import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig  # Add this import
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import argparse

def evaluate(model_path, test_file, results_dir):
    """
    Evaluates a fine-tuned model for text classification.
    Modified to handle PEFT models and your specific column names.
    """
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # Use your base model name
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data - MODIFIED FOR YOUR COLUMN NAMES
    test_data = pd.read_csv(test_file)
    if "body" not in test_data.columns or "label" not in test_data.columns:
        raise ValueError("Test data must contain 'body' and 'label' columns")
    
    # Combine subject and body for better context
    texts = (test_data["subject"].fillna("") + " " + test_data["body"]).tolist()
    y_true = test_data["label"].astype(int).values  # Ensure labels are integers

    # Tokenize
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(model.device)

     # Process in smaller batches
    batch_size = 4  # Start small, increase if you have more RAM
    predictions = []
    
    for i in tqdm(range(0, len(texts), desc="Evaluating")):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=512,  # Reduced from 2042 to 512
            return_tensors="pt",
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)
    
    y_pred = np.array(predictions)

    # Run predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        y_pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Generate and save metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(Path(results_dir) / "metrics.json", "w") as f:
        json.dump(report, f)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Phishing"])
    disp.plot()
    plt.savefig(Path(results_dir) / "confusion_matrix.png")
    plt.close()

    print(f"Evaluation complete! Results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    evaluate(args.model_path, args.test_file, args.results_dir)