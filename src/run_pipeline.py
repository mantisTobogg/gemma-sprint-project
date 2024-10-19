# run_pipeline.py (Run the Complete Pipeline)
import os
import pandas as pd
from .config import CONFIG  # Import CONFIG to access batch size
from .main import (
    analyze_sentiment, detect_sarcasm,
    contains_offensive_language, generate_responses
)
from .batch_processing import batch_process

# Load dataset
DATASET_PATH = os.path.join(os.getcwd(), "datasets", "UScomments.csv")
data = pd.read_csv(DATASET_PATH, on_bad_lines="skip")

if 'comment_text' not in data.columns:
    raise ValueError("The 'comment_text' column is missing from the dataset.")
comments = data['comment_text'].astype(str).tolist()

# Initialize lists to store results
sentiments, sarcasm_labels, offensive_flags = [], [], []

# Process sentiment in batches
for sentiment_batch in batch_process(comments, CONFIG["batch_size"], analyze_sentiment):
    sentiments.extend(sentiment_batch)

# Process sarcasm detection in batches
for sarcasm_batch in batch_process(comments, CONFIG["batch_size"], detect_sarcasm):
    sarcasm_labels.extend(sarcasm_batch)

# Detect offensive language
offensive_flags = [contains_offensive_language(comment) for comment in comments]

# Generate responses
responses = generate_responses(sentiments, comments)

# Combine results into a DataFrame
df_results = pd.DataFrame({
    "Comment": comments,
    "Sentiment": sentiments,
    "Sarcasm": sarcasm_labels,
    "Offensive_Flag": offensive_flags,
    "Response": responses
})

# Save results to CSV
OUTPUT_PATH = os.path.join(os.getcwd(), "outputs", "Processed_Comments.csv")
df_results.to_csv(OUTPUT_PATH, index=False)
print(f"Processing complete. Results saved to '{OUTPUT_PATH}'.")