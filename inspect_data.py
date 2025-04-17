import pandas as pd
from pathlib import Path

def inspect_phishing():
    try:
        df = pd.read_csv(Path("data/extracted/phishing/enron.csv"))
        print("\n=== PHISHING DATA STRUCTURE ===")
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        print("\nFirst 2 records:")
        print(df.head(2).to_markdown())
        
        # Check for text content
        sample_text = df.iloc[0].to_dict()
        print("\nSample record:")
        for k, v in sample_text.items():
            print(f"{k}: {str(v)[:100]}...")

    except Exception as e:
        print(f"Phishing data error: {e}")

def inspect_enron():
    try:
        enron_path = Path("data/extracted/enron/maildir")
        txt_files = list(enron_path.rglob("*.txt"))
        print(f"\n=== ENRON DATA STRUCTURE ===")
        print(f"Found {len(txt_files)} email files")
        
        # Show first file structure
        first_file = txt_files[0]
        print(f"\nFirst file: {first_file}")
        with open(first_file, 'r', encoding='latin1') as f:
            print("\nContent preview:")
            print(f.read()[:500] + "...")

    except Exception as e:
        print(f"Enron data error: {e}")

if __name__ == "__main__":
    inspect_phishing()
    inspect_enron()