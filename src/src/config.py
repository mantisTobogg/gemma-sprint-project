# Config.py
import torch
import logging
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# Set up logging configuration to support real-time logging and rotation
LOG_FILENAME = f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Basic configuration for logging to file
logging.basicConfig(
    level=logging.INFO,  # You can set to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        TimedRotatingFileHandler(
            LOG_FILENAME, when="H", interval=1, backupCount=24  # Rotate every hour
        ),
        logging.StreamHandler()  # StreamHandler to print logs to console
    ],
    force=True  # Force to override any prior log handlers
)

CONFIG = {
    "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 128,
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
