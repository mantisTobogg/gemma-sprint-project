# 4. run_pipeline.py – Main Execution Script

# 	•	Purpose: Orchestrates the entire pipeline by:
# 	•	Loading the dataset
# 	•	Running batch processing for sentiment and sarcasm analysis
# 	•	Generating responses for relevant comments
# 	•	Saving the results to CSV
# 	•	Equivalent Cells in Original .ipynb:
# 	•	Dataset loading and validation
# → Equivalent to Cell 2 (CSV loading and extraction of comment_text).
# 	•	Final aggregation of sentiment, sarcasm, and offensive detection results
# → Corresponds to the final summary and result display sections in the notebook.

# run_pipeline.py (Run the Complete Pipeline)
from main import (
    analyze_sentiment, detect_sarcasm, 
    contains_offensive_language, generate_responses
)
from batch_processing import batch_process
import pandas as pd

# Load dataset
data = pd.read_csv("UScomments.csv", on_bad_lines="skip")
comments = data['comment_text'].astype(str).tolist()

# Process in batches
sentiments = []
sarcasm_labels = []
offensive_flags = []

for sentiment_batch in batch_process(comments, CONFIG["batch_size"], analyze_sentiment):
    sentiments.extend(sentiment_batch)

for sarcasm_batch in batch_process(comments, CONFIG["batch_size"], detect_sarcasm):
    sarcasm_labels.extend(sarcasm_batch)

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

# Save results
df_results.to_csv("Processed_Comments.csv", index=False)
print("Processing complete. Results saved to 'Processed_Comments.csv'.")