# /src/evaluate.py
import json
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from transformers import pipeline
from pathlib import Path
from .config import MODEL_DIR, DATA_DIR
import matplotlib.pyplot as plt

def evaluate():
    # Load fine-tuned model
    classifier = pipeline(
        "text-classification",
        model=str(MODEL_DIR / "fine_tuned"),
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Load test data
    test_data = pd.read_csv(DATA_DIR / "processed/test.csv")
    
    # Run predictions
    preds = classifier(test_data["text"].tolist())
    y_pred = [1 if p["label"] == "LABEL_1" else 0 for p in preds]
    y_true = test_data["label"].values
    
    # Generate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(Path("results") / "metrics.json", "w") as f:
        json.dump(report, f)
    
    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(Path("results") / "confusion_matrix.png")

if __name__ == "__main__":
    evaluate()