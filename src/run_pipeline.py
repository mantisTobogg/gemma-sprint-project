# run_pipeline.py - Runs the sentiment and sarcasm analysis pipeline

import pandas as pd
from main import analyze_sentiment, detect_sarcasm, generate_gemma_response
from batch_processing import batch_process
from config import CONFIG

def run_pipeline(input_file):
    """Run the sentiment and sarcasm analysis on the input data."""
    data = pd.read_csv(input_file)
    comments = data['comment'].tolist()

    sentiment_labels = []
    sarcasm_labels = []

    # Perform Sentiment Analysis
    for sentiment_batch in batch_process(comments, CONFIG["batch_size"], analyze_sentiment):
        sentiment_labels.extend(sentiment_batch)

    # Perform Sarcasm Detection
    for sarcasm_batch in batch_process(comments, CONFIG["batch_size"], detect_sarcasm):
        sarcasm_labels.extend(sarcasm_batch)

    # Generate Gemma Responses
    gemma_responses = [generate_gemma_response(comment) for comment in comments]

    # Save Results
    results = pd.DataFrame({
        "Comment": comments,
        "Sentiment": sentiment_labels,
        "Sarcasm": sarcasm_labels,
        "Gemma Response": gemma_responses
    })

    results.to_csv("analysis_results.csv", index=False)
    print("Pipeline completed and results saved to analysis_results.csv.")