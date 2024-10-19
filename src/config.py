# config.py (Configuration Settings)
import torch

CONFIG = {
    "models": {
        "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        "sarcasm": "helinivan/english-sarcasm-detector",
        "response": "google/gemma-2-2b-it"
    },
    "thresholds": {
        "positive": 0.6,
        "negative": 0.4
    },
    "device": 0 if torch.cuda.is_available() else -1,
    "batch_size": 32  # Adjust based on memory availability
}

# 1. config.py – Model Configurations & Thresholds

#	•	Purpose: Store model names, thresholds, device settings, and batch size.
#	•	Equivalent Cells in Original .ipynb:
#	•	Cells where model names, thresholds, and device configurations were set (like distilbert-base-uncased, device detection with CUDA, etc.).