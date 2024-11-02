# minilm_integration/generate_term_embeddings.py

import os
import pandas as pd
import torch
from load_minilm import encode_text

# Load glossary terms
data_path = os.path.join("datasets", "IT_glossory_final_v4.csv")
df = pd.read_csv(data_path)

# Define output path for precomputed embeddings
output_path = os.path.join("minilm_integration", "term_embeddings.pt")

def generate_term_embeddings(df):
    """Generates embeddings for each glossary term."""
    term_embeddings = {}
    for _, row in df.iterrows():
        term = row['Term']  # Adjusted column name
        embedding = encode_text(term).cpu()
        term_embeddings[term] = embedding
    return term_embeddings

# Generate and save embeddings
if __name__ == "__main__":
    term_embeddings = generate_term_embeddings(df)
    torch.save(term_embeddings, output_path)
    print(f"Term embeddings saved to {output_path}")
