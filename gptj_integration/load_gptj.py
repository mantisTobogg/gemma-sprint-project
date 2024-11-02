# Loading and dowloading of model
from transformers import AutoTokenizer, AutoModelForCausalLM

# Existing code
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
