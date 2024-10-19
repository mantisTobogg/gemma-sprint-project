# main.py (Main Workflow)
import pandas as pd
from transformers import pipeline
from better_profanity import profanity
from .config import CONFIG

# Initialize models globally to avoid reloading them repeatedly
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=CONFIG["models"]["sentiment"],
    device=CONFIG["device"]
)

def analyze_sentiment(comments):
    return [res['label'] for res in sentiment_analyzer(comments)]

def detect_sarcasm(comments):
    sarcasm_detector = pipeline("text-classification", model=CONFIG["models"]["sarcasm"], device=CONFIG["device"])
    return [res['label'] for res in sarcasm_detector(comments)]

def contains_offensive_language(comment):
    return profanity.contains_profanity(comment)

def generate_responses(sentiments, comments):
    response_generator = pipeline("text-generation", model=CONFIG["models"]["response"])
    responses = []
    for sentiment, comment in zip(sentiments, comments):
        if sentiment != "Neutral":
            prompt = f"The following comment is {sentiment}: {comment}"
            response = response_generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
            responses.append(response)
        else:
            responses.append(None)
    return responses