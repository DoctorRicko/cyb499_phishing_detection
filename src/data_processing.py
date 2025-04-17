import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from .config import DATA_DIR, MODEL_NAME, MAX_LENGTH, TEST_SIZE

def load_and_prepare_data():
    """Load raw data and preprocess"""
    # Load your datasets (modify paths as needed)
    phish_df = pd.read_csv(DATA_DIR / "raw/phishtank.csv")
    legit_df = pd.read_csv(DATA_DIR / "raw/enron.csv")
    
    # Combine and shuffle
    df = pd.concat([
        phish_df.assign(label=1),
        legit_df.assign(label=0)
    ]).sample(frac=1)
    
    return df

def tokenize_dataset(df):
    """Convert to HF Dataset and tokenize"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_pandas(df)
    
    return dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        ),
        batched=True
    ).train_test_split(test_size=TEST_SIZE)