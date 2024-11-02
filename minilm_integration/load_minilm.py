# minilm_integration/load_minilm.py

import torch
from transformers import AutoModel, AutoTokenizer
import os
from torch.nn.functional import cosine_similarity

# Model name and device configuration
MINILM_MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load MiniLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL_NAME)
model = AutoModel.from_pretrained(MINILM_MODEL_NAME).to(device)

# Load precomputed term embeddings
embedding_path = os.path.join("minilm_integration", "term_embeddings.pt")
term_embeddings = torch.load(embedding_path, map_location=device, weights_only=True)

def encode_text(text, min_length=20, max_length=128):
    """Encodes input text into embeddings using MiniLM with specified min and max length."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)

    # Adjust minimum length
    input_ids = inputs['input_ids']
    if input_ids.size(1) < min_length:
        pad_length = min_length - input_ids.size(1)
        padding = torch.full((input_ids.size(0), pad_length), tokenizer.pad_token_id).to(device)
        inputs['input_ids'] = torch.cat([input_ids, padding], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(padding)], dim=1)

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def find_closest_term(query):
    """Finds the closest term in the glossary for a given query."""
    query_embedding = encode_text(query).cpu().view(1, -1)  # Ensure shape is (1, 384)

    best_match, max_similarity = None, -1
    for term, term_embedding in term_embeddings.items():
        term_embedding = term_embedding.to(query_embedding.device).view(1, -1)  # Reshape to (1, 384) for consistency
        similarity = cosine_similarity(query_embedding, term_embedding).item()  # Calculate scalar similarity
        if similarity > max_similarity:
            best_match, max_similarity = term, similarity

    return best_match if max_similarity > 0.5 else None  # Set threshold for similarity
