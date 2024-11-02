import os
import pandas as pd
from transformers import GPTJForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from tqdm import tqdm

data_path = 'datasets/IT_glossory_final_v4.csv'

# Load Dataset
df = pd.read_csv(data_path)

class GlossaryDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # Important for training to match labels
        }

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# Disable use_cache to be compatible with gradient checkpointing
model.config.use_cache = False

# Prepare Dataset
definitions = df['Definition'].tolist()
train_texts, val_texts = train_test_split(definitions, test_size=0.1, random_state=42)

train_dataset = GlossaryDataset(train_texts, tokenizer, max_length=256)
val_dataset = GlossaryDataset(val_texts, tokenizer, max_length=256)

# Training Arguments
training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=2,
    gradient_checkpointing=True,  # Turn off gradient checkpointing
    remove_unused_columns=False,
    bf16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()
