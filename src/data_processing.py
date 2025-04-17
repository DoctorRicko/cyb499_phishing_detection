import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from config import DATA_DIR, MAX_LENGTH, MODEL_NAME  # Import from config

def load_enron_emails():
    """Load legitimate emails (label=0)"""
    df = pd.read_csv(DATA_DIR / "extracted/phishing/enron.csv")
    legit = df[df['label'] == 0]
    texts = legit['subject'].fillna('') + "\n\n" + legit['body'].fillna('')
    return pd.DataFrame({'text': texts, 'label': 0})

def load_phishing_data():
    """Load phishing emails (label=1) with keyword filtering"""
    df = pd.read_csv(DATA_DIR / "extracted/phishing/enron.csv")
    phish = df[df['label'] == 1]
    
    # Combine subject + body and filter for suspicious keywords
    texts = phish['subject'].fillna('') + "\n\n" + phish['body'].fillna('')
    keywords = ['click', 'verify', 'account', 'password', 'urgent']
    mask = texts.str.contains('|'.join(keywords), case=False)
    
    return pd.DataFrame({'text': texts[mask], 'label': 1})

def prepare_datasets():
    """Main dataset preparation with balanced classes"""
    legit_df = load_enron_emails()
    phish_df = load_phishing_data()
    
    # Balance classes
    min_samples = min(len(legit_df), len(phish_df))
    combined_df = pd.concat([
        legit_df.sample(min_samples, random_state=42),
        phish_df.sample(min_samples, random_state=42)
    ]).sample(frac=1, random_state=42)
    
    print(f"\nDataset Stats:")
    print(f"- Total samples: {len(combined_df)}")
    print(f"- Class balance:\n{combined_df['label'].value_counts()}")
    print(f"- Avg text length: {combined_df['text'].str.len().mean():.0f} chars")
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = Dataset.from_pandas(combined_df)
    
    return dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ),
        batched=True
    ).train_test_split(test_size=0.2)