# gemma_v5.py
import sys
import os
import random
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to sys.path for cross-directory imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import term recognition and sentiment classifier with correct paths
from minilm_integration.load_minilm import find_closest_term
from src.sentiment_model import SentimentClassifier

# Step 1: Templates for Each Sentiment
templates_positive = [
    "Thank you for your positive feedback! It's always encouraging to know our efforts make a difference in your technical journey.",
    "We're thrilled that our solutions are helping you navigate your IT challenges effectively. Your support means a lot to us!",
    "We appreciate your positive comments! If you need any further clarification on our recommendations, feel free to reach out.",
    "It's great to hear that our insights align well with your expectations. We're always here for additional support.",
    "Thank you for your positive response. Our aim is to provide you with actionable technical advice, and we're glad it's working for you.",
    "We're glad to see that our recommendations helped! Let us know if there’s anything else we can assist with in your project.",
]

templates_negative = [
    "We understand your concerns regarding the infrastructure compatibility. It’s crucial to address such issues thoroughly.",
    "Thank you for bringing this to our attention. Let's work through the compatibility questions regarding Oracle EBS and SAP HANA step by step.",
    "We sincerely apologize if our previous suggestions were unclear. Let’s revisit the integration details together.",
    "It sounds like there are some complex compatibility issues. We’ll help you untangle these challenges, especially concerning different systems.",
    "We understand the frustration that incompatibility concerns may cause. Let's clarify and explore possible solutions.",
    "Thank you for pointing out these issues. Let’s discuss the specifics to see how we can make this integration work for you.",
]

templates_sarcasm = [
    "We understand the underlying concern, and sarcasm aside, let’s address it directly: We'll guide you through how Oracle EBS aligns with SAP HANA.",
    "We appreciate the humor, but more importantly, we value getting to the root of your query. Let’s clarify the compatibility details.",
    "Noted, sarcasm and all. Let's take a direct approach to explore the ERP solution and address your concerns.",
    "It's clear there are some frustrations. Sarcasm aside, we're committed to working through the details with you.",
    "Point taken. Let's shift from humor to problem-solving: How can we make your Oracle EBS integration process easier?",
    "We get it—it’s complicated. Humor is a great coping mechanism, but let’s dive into the details and make it simpler.",
]

# Step 2: Continuation Prompts for Incomplete Responses
continuation_prompts = [
    "Would you like to explore this topic further?",
    "Shall we dive deeper into this subject?",
    "Would you like additional details on this?",
    "Is there anything more specific you’d like to know?",
    "Are you interested in a more in-depth explanation?",
    "Let me know if you want to continue with this topic!",
    "Can I provide more information on this subject?",
    "I can continue if you’d like to know more!",
    "Would you like me to elaborate further?",
    "Let’s keep going if you’re interested in more details!"
]

# Function to select a continuation prompt
def select_continuation_prompt():
    return random.choice(continuation_prompts)

# Step 3: Check for Close Matches if Term is Not Recognized
def handle_near_miss(user_comment):
    closest_term = find_closest_term(user_comment)
    if closest_term:
        return f"Are you perhaps referring to '{closest_term}'?"
    else:
        return "I'm not certain about that term. Could you clarify?"

# Step 4: Generate Response with Continuation Logic
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = model.to(device)

DEFAULT_NUM_BEAMS = 3
DEFAULT_MAX_TOKENS_BASE = 150
DEFAULT_MAX_TOKENS_MIN = 100
DEFAULT_MAX_TOKENS_MAX = 200
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_DO_SAMPLE = True

def generate_response(template, sentiment, num_beams=DEFAULT_NUM_BEAMS, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, do_sample=DEFAULT_DO_SAMPLE):
    max_tokens = (
        DEFAULT_MAX_TOKENS_MAX if len(template) > 150 or sentiment in [0, 1]
        else DEFAULT_MAX_TOKENS_MIN if len(template) < 50 and sentiment == 2
        else DEFAULT_MAX_TOKENS_BASE
    )

    input_ids = tokenizer.encode(template, return_tensors='pt').to(device)
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + max_tokens,
        num_beams=num_beams,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # If the response ends abruptly, add a continuation prompt
    if response.endswith(",") or response.endswith("and"):
        response += " " + select_continuation_prompt()

    return response

# Step 5: Select Template Function
def select_template(sentiment):
    if sentiment == 2:  # Positive
        return random.choice(templates_positive)
    elif sentiment == 0:  # Negative
        return random.choice(templates_negative)
    elif sentiment == 1:  # Sarcasm
        return random.choice(templates_sarcasm)
    else:
        raise ValueError("Invalid sentiment")

# Step 6: Integrate Term Recognition and Sentiment Analysis for Response Generation
def handle_user_comment(user_comment):
    recognized_term = find_closest_term(user_comment)
    
    if recognized_term:
        response_template = f"Here's some information about {recognized_term}: "
    else:
        # Check for a close match and prompt if available
        near_miss_prompt = handle_near_miss(user_comment)
        return near_miss_prompt

    sentiment_model = SentimentClassifier(n_classes=3)
    state_dict = torch.load('checkpoints/sentiment_model_after_epoch_1.pth', map_location=device, weights_only=True)
    sentiment_model.load_state_dict(state_dict)  # Correct usage of torch.load
    sentiment_model.eval()
    sentiment_model = sentiment_model.to(device)

    tokenizer_sentiment = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    encoding = tokenizer_sentiment.encode_plus(
        user_comment,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device).long()
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        sentiment = torch.argmax(sentiment_model(input_ids, attention_mask), dim=1).item()

    response_template = select_template(sentiment)
    complete_response = generate_response(response_template, sentiment)
    return complete_response

# Example Usage
if __name__ == "__main__":
    user_comments = [
        "Can you explain ABAP in SAP?",
        "What do you think about Oracle compatibility?",
        "Is there a better way to handle data integration for ERP systems?",
        "How does SAP HANA differ from traditional databases?",
        "Why is my ERP setup slow?"
    ]

    for comment in user_comments:
        print(f"User Comment: {comment}")
        response = handle_user_comment(comment)
        print(f"Response: {response}\n")
