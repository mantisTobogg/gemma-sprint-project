import logging  # Logging enabled
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from config import CONFIG  # Import CONFIG

# Initialize logging
logging.basicConfig(
    filename='logs/main.log',  # Log file path
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Setup device: MPS, CUDA, or CPU
device = torch.device(CONFIG["device"])

# Load the sarcasm detection model and tokenizer
logging.info("Loading sarcasm detection model and tokenizer...")
sarcasm_tokenizer = AutoTokenizer.from_pretrained(CONFIG["models"]["sarcasm"])
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(CONFIG["models"]["sarcasm"]).to(device)

# Sarcasm Detection Function
def detect_sarcasm(comment):
    """Detect sarcasm in a single comment."""
    logging.info(f"Detecting sarcasm for a comment: {comment}")

    try:
        inputs = sarcasm_tokenizer(
            comment,
            return_tensors="pt",
            padding=True,  # Ensure consistent input length
            truncation=True,  # Avoid exceeding max length
            max_length=128  # Control input size
        ).to(device)

        outputs = sarcasm_model(**inputs)
        sarcasm_score = outputs.logits.softmax(dim=1)[0][1].item()
        logging.info(f"Sarcasm score: {sarcasm_score}")
        return sarcasm_score

    except Exception as e:
        logging.error(f"Error detecting sarcasm: {e}")
        raise  # Re-raise to ensure visibility

# GEMMA Text Generation
def generate_gemma_response(prompt):
    """Generate response using GEMMA model."""
    logging.info(f"Generating GEMMA response for prompt: {prompt}")

    try:
        generator = pipeline(
            "text-generation",
            model=CONFIG["models"]["text_generation"],
            device=device
        )

        response = generator(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
        logging.info(f"GEMMA generated response: {response}")
        return response

    except Exception as e:
        logging.error(f"Error generating GEMMA response: {e}")
        raise  # Re-raise to ensure visibility
