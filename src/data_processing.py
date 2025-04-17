import os
import tarfile
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from .config import DATA_DIR, MODEL_NAME, MAX_LENGTH

def load_enron_emails():
    """Process Enron maildir structure"""
    enron_path = DATA_DIR / "extracted/enron/maildir"
    emails = []
    
    for root, _, files in os.walk(enron_path):
        for file in files:
            if file.endswith('.txt'):
                with open(Path(root)/file, 'r', encoding='latin1') as f:
                    try:
                        content = f.read()
                        emails.append({"text": content, "label": 0})
                    except UnicodeDecodeError:
                        continue
    return pd.DataFrame(emails).sample(frac=1)

def load_phishing_data():
    """Load phishing CSV"""
    df = pd.read_csv(DATA_DIR / "extracted/phishing/enron.csv")
    return df[['text']].assign(label=1)  # Assuming column named 'text'

def prepare_datasets():
    # Load both datasets
    legit_df = load_enron_emails()
    phish_df = load_phishing_data()
    
    # Combine and shuffle
    combined = pd.concat([legit_df, phish_df]).sample(frac=1)
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_pandas(combined)
    
    return dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        ),
        batched=True
    ).train_test_split(test_size=0.2)