import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# Load dataset
df = pd.read_csv('datasets/IT_glossory_final_v4.csv')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize in batches
batch_size = 64
tokenized_texts = []

for i in tqdm(range(0, len(df), batch_size)):
    batch = df['Definition'].iloc[i:i + batch_size].tolist()
    tokenized_batch = tokenizer(
        batch,
        padding=True,  # Pad to the longest sequence in the batch
        truncation=True,  # Truncate to the model's max length
        return_tensors="pt",  # Return as PyTorch tensors
    )
    tokenized_texts.append(tokenized_batch)

# Save or further process `tokenized_texts` as needed
