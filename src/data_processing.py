from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from .config import DATA_DIR, MODEL_NAME, MAX_LENGTH

def load_datasets():
    """Load and merge phishing/legitimate datasets"""
    phish_df = pd.read_csv(DATA_DIR / "raw/phishtank/samples.csv")
    legit_df = pd.read_csv(DATA_DIR / "raw/enron/emails.csv")
    
    # Create labeled dataset
    df = pd.concat([
        phish_df[["text"]].assign(label=1),
        legit_df[["text"]].assign(label=0)
    ]).sample(frac=1)  # Shuffle
    
    return Dataset.from_pandas(df)

def tokenize_data(dataset):
    """Tokenize text data for LLM input"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH),
        batched=True
    )

def prepare_data():
    """Full data processing pipeline"""
    dataset = load_datasets()
    tokenized = tokenize_data(dataset)
    return tokenized.train_test_split(test_size=0.2)