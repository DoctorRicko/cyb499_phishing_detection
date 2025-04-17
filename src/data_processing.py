import os
import tarfile
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from .config import DATA_DIR, MAX_LENGTH

def load_phishing_data():
    """Load and preprocess your specific phishing CSV"""
    df = pd.read_csv(DATA_DIR / "extracted/phishing/enron.csv")

    # Create clean text by combining subject and body
    df['clean_text'] = df['subject'].fillna('') + "\n\n" + df['body'].fillna('')

    # Filter only phishing emails (label=1) if mixed dataset
    if 'label' in df.columns:
        df = df[df['label'] == 1]  # Use only phishing samples

    return df[['clean_text']].rename(columns={'clean_text': 'text'})

def load_enron_emails():
    """Process legitimate emails from Enron"""
    emails = []
    enron_path = DATA_DIR / "extracted/enron/maildir"

    for email_file in enron_path.rglob("*.txt"):
        with open(email_file, 'r', encoding='latin1') as f:
            try:
                content = f.read()
                # Extract body after first empty line (standard email format)
                body = content.split('\n\n', 1)[-1]
                emails.append({
                    'text': body,
                    'label': 0  # Mark as legitimate
                })
            except UnicodeDecodeError:
                continue

    return pd.DataFrame(emails).sample(frac=1)

def prepare_datasets(model_name): # Accept model_name as an argument
    """Prepare dataset with your specific structure"""
    # Load data
    legit_df = load_enron_emails()
    phish_df = load_phishing_data()

    # Verify counts
    print(f"Loaded {len(legit_df)} legitimate emails")
    print(f"Loaded {len(phish_df)} phishing emails")

    # Combine and shuffle
    combined = pd.concat([legit_df.assign(label=0), phish_df.assign(label=1)]).sample(frac=1).reset_index(drop=True)

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_name) # Use the passed model_name
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