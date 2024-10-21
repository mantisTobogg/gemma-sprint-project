# run_pipeline.py
import pandas as pd
from main import analyze_sentiment, detect_sarcasm, generate_gemma_response
from batch_processing import batch_process
from config import CONFIG
from concurrent.futures import ProcessPoolExecutor
import logging

# Initialize logging
logging.info("Starting pipeline execution...")

def run_pipeline(input_file):
    """Run the sentiment and sarcasm analysis on the input data."""
    # Load the input file
    try:
        data = pd.read_csv(input_file)
        comments = data['comment'].astype(str).tolist()  # Ensure all comments are strings
    except FileNotFoundError:
        logging.error(f"Input file '{input_file}' not found.")
        return
    except KeyError:
        logging.error("Expected column 'comment' not found in the input file.")
        return

    sentiment_labels = []
    sarcasm_labels = []

    # Perform Sentiment Analysis in Batches
    try:
        for sentiment_batch in batch_process(comments, CONFIG["batch_size"], analyze_sentiment):
            sentiment_labels.extend(sentiment_batch)
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")

    # Perform Sarcasm Detection in Batches
    try:
        for sarcasm_batch in batch_process(comments, CONFIG["batch_size"], detect_sarcasm):
            sarcasm_labels.extend(sarcasm_batch)
    except Exception as e:
        logging.error(f"Error during sarcasm detection: {e}")

    # Ensure that the sentiment and sarcasm labels match the number of comments
    assert len(sentiment_labels) == len(sarcasm_labels) == len(comments), "Mismatch in lengths after sentiment and sarcasm analysis."

    # Generate Gemma Responses in Parallel
    try:
        with ProcessPoolExecutor(max_workers=2) as executor:
            gemma_responses = list(executor.map(generate_gemma_response, comments))
    except Exception as e:
        logging.error(f"Error generating GEMMA responses: {e}")
        gemma_responses = [""] * len(comments)  # Fallback to empty responses if errors occur

    # Save Results
    results = pd.DataFrame({
        "Comment": comments,
        "Sentiment": sentiment_labels,
        "Sarcasm": sarcasm_labels,
        "Gemma Response": gemma_responses
    })

    # Save the DataFrame to a CSV
    results.to_csv("analysis_results.csv", index=False)
    logging.info("Pipeline completed. Results saved to 'analysis_results.csv'.")
    print("Pipeline completed and results saved to analysis_results.csv.")

if __name__ == "__main__":
    run_pipeline(CONFIG["datasets"]["cleaned"])