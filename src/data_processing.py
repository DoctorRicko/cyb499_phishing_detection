import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_LENGTH = 256

def load_data():
    """Load all data from the single CSV file using labels"""
    df = pd.read_csv(DATA_DIR / "extracted/phishing/enron.csv")
    
    # Separate legitimate (0) and phishing (1) emails
    legit = df[df['label'] == 0]
    phish = df[df['label'] == 1]
    
    # Combine subject and body for both types
    legit_texts = legit['subject'].fillna('') + "\n\n" + legit['body'].fillna('')
    phish_texts = phish['subject'].fillna('') + "\n\n" + phish['body'].fillna('')
    
    # Create labeled DataFrames
    legit_df = pd.DataFrame({'text': legit_texts, 'label': 0})
    phish_df = pd.DataFrame({'text': phish_texts, 'label': 1})
    
    return legit_df, phish_df

def prepare_datasets():
    """Prepare dataset using only the CSV data"""
    legit, phish = load_data()
    
    print(f"Found {len(legit)} legitimate and {len(phish)} phishing emails")
    
    # Balance classes
    min_count = min(len(legit), len(phish))
    combined = pd.concat([
        legit.sample(min_count, random_state=42),
        phish.sample(min_count, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    print("\nSample legitimate:")
    print(combined[combined['label']==0]['text'].iloc[0][:200] + "...")
    print("\nSample phishing:")
    print(combined[combined['label']==1]['text'].iloc[0][:200] + "...")
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
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