# config.py
import torch
import logging
from datetime import datetime

# Set up logging configuration
LOG_FILENAME = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)

import torch  # Ensure torch is imported

CONFIG = {
    "device": "mps" if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available() else
              "cpu",
    "batch_size": 32,
    "models": {
        "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        "sarcasm": "helinivan/english-sarcasm-detector",
        "text_generation": "google/gemma-2-2b-it"
    },
    "datasets": {
        "original": "datasets/UScomments.csv",
        "cleaned": "datasets/UScomments_final_cleaned.csv"
    },
    "outputs": {
        "processed": "outputs/Processed_Comments.csv"
    }
}
