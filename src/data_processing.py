import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from pathlib import Path

# Use absolute paths
DATA_DIR = Path(__file__).parent.parent / "data"
MAX_LENGTH = 256

def load_enron_emails():
    emails = []
    enron_path = DATA_DIR / "extracted/enron/maildir"
    for email_file in enron_path.glob("**/*.txt"):
        with open(email_file, "r", encoding="latin1") as f:
            emails.append({
                "text": f.read(),
                "label": 0
            })
    return pd.DataFrame(emails)

def load_phishing_data():
    """Load phishing emails from your specific CSV structure"""
    df = pd.read_csv(DATA_DIR / "extracted/phishing/enron.csv")
    
    # Combine subject and body with clear separation
    texts = df['subject'].fillna('') + "\n\n" + df['body'].fillna('')
    
    # Only use samples marked as phishing (label=1)
    if 'label' in df.columns:
        return pd.DataFrame({
            'text': texts[df['label'] == 1],  # Filter for phishing only
            'label': 1
        })
    else:
        return pd.DataFrame({'text': texts, 'label': 1})

def prepare_datasets():
    legit = load_enron_emails()
    phish = load_phishing_data()
    
    # Balance classes
    min_samples = min(len(legit), len(phish))
    combined = pd.concat([
        legit.sample(min_samples),
        phish.sample(min_samples)
    ])
    
    # Tokenize
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