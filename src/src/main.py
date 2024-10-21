import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config import CONFIG, LOG_FILENAME
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os

# Constants
MAX_LENGTH = 128
MAX_WORKERS = 8  # Adjust to suit hardware capability
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
CHECKPOINT_FILE = '/Users/kihun/Documents/gemma_NON-GIT files/gemma-sprint-project/checkpoints/sarcasm_labels.pkl'

# Initialize logging
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=LOG_FORMAT)

# Setup device: MPS, CUDA, or CPU
device = torch.device(CONFIG.get("device", "cpu"))


# Lazy model loading functions
def load_model(tokenizer_model_name=None, model_name=None, task=None):
    logging.info(f"Loading {task or 'model'}...")

    if task == "text-generation":
        if model_name is None:
            raise ValueError("Model name must be specified for text generation.")
        return pipeline(
            "text-generation",
            model=CONFIG["models"][model_name],
            device=0 if device.type == "cuda" else -1
        )

    if tokenizer_model_name and model_name:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["models"][tokenizer_model_name])
        model = AutoModelForSequenceClassification.from_pretrained(CONFIG["models"][model_name]).to(device)
        return tokenizer, model

    logging.error("Error: Missing required arguments for model loading.")
    raise ValueError("Missing required arguments for model loading.")


# GEMMA Text Generation Function
def generate_gemma_response(prompt, sarcasm_flag, max_new_tokens=40):
    logging.info(f"Generating GEMMA response for prompt: {prompt}, sarcasm_flag: {sarcasm_flag}")
    try:
        if 'generator' not in globals():
            global generator
            generator = load_model(model_name="text_generation", task="text-generation")

        # Modify prompt if sarcasm is detected
        if sarcasm_flag > 0.5:
            prompt = f"Provide a witty and sarcastic response to this comment: '{prompt}'"

        response = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)[0]["generated_text"]
        logging.info(f"GEMMA generated response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error generating GEMMA response for prompt '{prompt}': {e}")
        return ""


# Sarcasm Detection Function
def detect_sarcasm(comment):
    logging.info(f"Detecting sarcasm for comment: {comment}")
    try:
        if 'sarcasm_tokenizer' not in globals() or 'sarcasm_model' not in globals():
            global sarcasm_tokenizer, sarcasm_model
            sarcasm_tokenizer, sarcasm_model = load_model("sarcasm", "sarcasm")
        inputs = sarcasm_tokenizer(comment, return_tensors="pt", padding=True, truncation=True,
                                   max_length=MAX_LENGTH).to(device)
        outputs = sarcasm_model(**inputs)
        sarcasm_score = outputs.logits.softmax(dim=1)[0][1].item()
        logging.info(f"Sarcasm score: {sarcasm_score}")
        return sarcasm_score
    except Exception as e:
        logging.error(f"Error detecting sarcasm in comment '{comment}': {e}")
        return 0.0


# Sentiment Analysis Function
def analyze_sentiment(comment):
    logging.info(f"Analyzing sentiment for comment: {comment}")
    try:
        if 'sentiment_tokenizer' not in globals() or 'sentiment_model' not in globals():
            global sentiment_tokenizer, sentiment_model
            sentiment_tokenizer, sentiment_model = load_model("sentiment", "sentiment")
        inputs = sentiment_tokenizer(comment, return_tensors="pt", padding=True, truncation=True,
                                     max_length=MAX_LENGTH).to(device)
        outputs = sentiment_model(**inputs)
        sentiment_score = outputs.logits.softmax(dim=1)[0]
        sentiment_label = torch.argmax(sentiment_score).item()
        sentiment_label_str = "POSITIVE" if sentiment_label == 1 else "NEGATIVE"
        logging.info(
            f"Sentiment analysis result for comment '{comment}': {sentiment_label_str} with score: {sentiment_score[sentiment_label].item()}")
        return sentiment_label_str, sentiment_score[sentiment_label].item()
    except Exception as e:
        logging.error(f"Error analyzing sentiment for comment '{comment}': {e}")
        return "Unknown", 0.0


# Process Comment Function
def process_comment(comment):
    try:
        sarcasm_flag = detect_sarcasm(comment)
        response = generate_gemma_response(comment, sarcasm_flag)
        sentiment_label, sentiment_score = analyze_sentiment(comment)
        return {
            "comment": comment,
            "sarcasm_flag": sarcasm_flag,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "gemma_response": response
        }
    except Exception as e:
        logging.error(f"Error processing comment '{comment}': {str(e)}")
        return {
            "comment": comment,
            "sarcasm_flag": False,
            "sentiment_label": "Unknown",
            "sentiment_score": 0.0,
            "gemma_response": ""
        }


# Main execution
if __name__ == "__main__":
    # Step to load previously computed sarcasm labels if available
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            sarcasm_labels = pickle.load(f)
        logging.info(f"Loaded {len(sarcasm_labels)} sarcasm labels from checkpoint file.")
    except FileNotFoundError:
        sarcasm_labels = []
        logging.info("No checkpoint found. Proceeding without precomputed sarcasm labels.")

    # Load comments (e.g., from your dataset or another source)
    # Assuming 'comments' variable has already been initialized in the script
    comments = ["This is the first comment.", "Another sarcastic comment for testing."]  # Replace with actual data
    processed_comments = []

    # Calculate how many comments have not yet been processed
    unprocessed_comments = comments[len(sarcasm_labels):]

    # Using tqdm to track progress in generating responses
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_comment = {executor.submit(process_comment, comment): comment for comment in unprocessed_comments}
        for future in tqdm(as_completed(future_to_comment), total=len(unprocessed_comments), desc="Processing Comments"):
            try:
                result = future.result()
                processed_comments.append(result)
            except Exception as e:
                logging.error(f"Error processing comment in parallel: {e}")

    # Combine the old labels with new results if there were any unprocessed comments
    sarcasm_labels.extend([result["sarcasm_flag"] for result in processed_comments])

    # Save the updated checkpoint file
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(sarcasm_labels, f)
    logging.info(f"Updated sarcasm labels saved. Total labels now: {len(sarcasm_labels)}")

    # Print the results
    for result in processed_comments:
        print(result)
