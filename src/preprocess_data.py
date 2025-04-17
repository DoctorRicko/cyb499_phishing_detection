from src.data_processing import prepare_datasets
from src.config import PROCESSED_DIR
import pandas as pd

if __name__ == "__main__":
    dataset = prepare_datasets()
    
    # Save processed data
    PROCESSED_DIR.mkdir(exist_ok=True)
    dataset["train"].to_pandas().to_csv(PROCESSED_DIR/"train.csv", index=False)
    dataset["test"].to_pandas().to_csv(PROCESSED_DIR/"test.csv", index=False)