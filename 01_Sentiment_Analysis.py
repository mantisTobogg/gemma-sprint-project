#!/usr/bin/env python
# coding: utf-8

# Step A: Add the 'src' directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Step B: Import necessary modules
import torch
import logging
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import shutil

# Import configuration and custom functions
from config import CONFIG
from main import analyze_sentiment, detect_sarcasm, generate_gemma_response
from batch_processing import batch_process

# Now the script is formatted properly and ready for execution

# Constants
MAX_WORKERS = 8  # Adjust to suit hardware capability
CHECKPOINT_FILE = 'checkpoints/sarcasm_labels.pkl'
BACKUP_FILE = 'checkpoints/sarcasm_labels_backup.pkl'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# Setup device: MPS, CUDA, or CPU
device = torch.device(CONFIG.get("device", "cpu"))

# Step 1: Define the path to the cleaned dataset
DATASET_PATH = os.path.join(os.getcwd(), "datasets", "UScomments_final_cleaned.csv")

# Step 2: Load the cleaned dataset
data = pd.read_csv(DATASET_PATH, low_memory=False)
logging.info(f"Loaded dataset from {DATASET_PATH}")

# Step 3: Extract comments and pre-labeled sentiment from the dataset
comments = data["cleaned_comment"].astype(str).tolist()
pre_labeled_sentiments = data["sentiment"].tolist()

# Step 4: Initialize a list to store sarcasm detection results
sarcasm_labels = []

# Step 5: Perform sarcasm detection in batches with a progress bar
# Create checkpoint directory if it doesn't exist
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Ensure write permission to the checkpoint directory
if not os.access(checkpoint_dir, os.W_OK):
    logging.warning(f"No write permission for checkpoint directory: {checkpoint_dir}")

# Backup the current checkpoint if exists
if os.path.exists(CHECKPOINT_FILE):
    shutil.copy2(CHECKPOINT_FILE, BACKUP_FILE)
    logging.info(f"Backup created at: {BACKUP_FILE}")

# Load sarcasm checkpoint if available
try:
    with open(CHECKPOINT_FILE, 'rb') as f:
        sarcasm_labels = pickle.load(f)
    logging.info("Loaded sarcasm labels from checkpoint file.")
except FileNotFoundError:
    logging.info("No checkpoint found. Proceeding without precomputed sarcasm labels.")

# Sarcasm detection batch process
total_batches = len(comments) // CONFIG["batch_size"] + (1 if len(comments) % CONFIG["batch_size"] != 0 else 0)
for idx, sarcasm_batch in enumerate(tqdm(batch_process(comments, CONFIG["batch_size"], detect_sarcasm), total=total_batches, desc="Performing Sarcasm Detection")):
    sarcasm_labels.extend(sarcasm_batch)
    logging.info(f"Processed batch {idx + 1}/{total_batches} in sarcasm detection.")
    # Save checkpoint after each batch
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(sarcasm_labels, f)
logging.info("Step 5: Sarcasm detection completed and checkpoint saved.")

# Step 6: Generate responses using the Gemma-2b-it model (parallelized)
responses = []  # List to hold the generated responses

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_comment = {executor.submit(generate_gemma_response, comment, sarcasm_flag): comment for comment, sarcasm_flag in zip(comments, sarcasm_labels)}
    for future in tqdm(as_completed(future_to_comment), total=len(comments), desc="Generating GEMMA Responses"):
        try:
            response = future.result()
            responses.append(response)
        except Exception as e:
            logging.error(f"Error generating GEMMA response in parallel: {e}")
            responses.append("")

logging.info("Step 6: GEMMA response generation completed.")

# Step 7: Combine all results into a DataFrame
assert len(comments) == len(pre_labeled_sentiments) == len(sarcasm_labels) == len(responses), "List lengths do not match. Check the previous steps for errors."

df_results = pd.DataFrame({
    "Comment": comments,
    "Sentiment": pre_labeled_sentiments,
    "Sarcasm": sarcasm_labels,
    "Response": responses
})
logging.info(f"Step 7: Combined results into DataFrame with shape {df_results.shape}.")

# Step 8: Save the results to a CSV file
OUTPUT_PATH = os.path.join(os.getcwd(), "outputs", "Processed_Comments.csv")
df_results.to_csv(OUTPUT_PATH, index=False)
logging.info(f"Step 8: Results saved to '{OUTPUT_PATH}'.")

# Step 9: Display a sample of the final results
df_sample = df_results.head()
print(df_sample)

# Monitor logs in real-time if needed (this part can be commented out if not required)
logs_directory = os.path.join(os.getcwd(), "logs")
log_files = glob.glob(os.path.join(logs_directory, "*.log"))
if log_files:
    latest_log_file = max(log_files, key=os.path.getctime)
    print(f"Monitoring latest log file: {latest_log_file}")
    safe_log_file_path = f'"{latest_log_file}"'
    os.system(f'tail -f {safe_log_file_path}')
else:
    print("No log files found in the logs directory.")

