# 2. main.py – Core Logic & Functions

# 	•	Purpose: Contains all essential functions for:
# 	•	Sentiment Analysis (analyze_sentiment)
# 	•	Sarcasm Detection (detect_sarcasm)
# 	•	Offensive Language Detection (contains_offensive_language)
# 	•	Response Generation (generate_responses)
# 	•	Equivalent Cells in Original .ipynb:
# 	•	Sentiment analysis (using Hugging Face transformers)
# → Equivalent to Cell 3 in your notebook.
# 	•	Offensive language detection (with better-profanity)
# → Equivalent to Cell A (extended offensive language detection).
# 	•	Helper functions to refine labels
# → Comes from Cell 1 (refine_labels).
# 	•	Response generation using gemma-2b-it
# → Matches with Gemma-2b-it integration section from your notebook.


# main.py (Main Workflow)
import pandas as pd
from transformers import pipeline
from better_profanity import profanity
from config import CONFIG

# Initialize models globally to avoid reloading them repeatedly
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model=CONFIG["models"]["sentiment"], 
    device=CONFIG["device"]
)
sarcasm_detector = pipeline(
    "text-classification", 
    model=CONFIG["models"]["sarcasm"], 
    device=CONFIG["device"]
)

def refine_labels(score):
    """Refines sentiment based on score thresholds."""
    if score > CONFIG["thresholds"]["positive"]:
        return "Positive"
    elif score < CONFIG["thresholds"]["negative"]:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment(comments):
    """Analyze sentiment for a list of comments."""
    results = sentiment_analyzer(comments)
    return [refine_labels(res['score']) for res in results]

def detect_sarcasm(comments):
    """Detect sarcasm in a list of comments."""
    return [res['label'] for res in sarcasm_detector(comments)]

def contains_offensive_language(comment):
    """Check if a comment contains offensive language."""
    return profanity.contains_profanity(comment)

def generate_responses(sentiments, comments):
    """Generate responses for non-neutral comments."""
    from transformers import pipeline
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