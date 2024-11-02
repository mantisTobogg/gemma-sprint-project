import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel

# Step 1: Define Dataset Class
class CommentsDataset(Dataset):
    def __init__(self, csv_file):
        # Load dataset from CSV file
        self.data = pd.read_csv(csv_file)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract comment text and label
        comment = self.data.iloc[idx]['cleaned_comment']
        label = self.data.iloc[idx]['sentiment']

        # Tokenize the comment
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 2: Define the Sentiment Analysis Model
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # Pass inputs through DistilBERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Extract the last hidden state for [CLS] token
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.out(output)

# Step 3: Training Function

def train_model(model, data_loader, loss_fn, optimizer, device, n_epochs):
    model = model.to(device)
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0
        for i, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}')

        # Save checkpoint after each epoch
        torch.save(model.state_dict(), f'checkpoints/sentiment_model_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved for epoch {epoch+1}.")

# Step 4: Main Script for Setting Up and Training
if __name__ == "__main__":
    # Load dataset
    dataset = CommentsDataset(csv_file='datasets/UScomments_final_cleaned.csv')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SentimentClassifier(n_classes=3)  # 3 classes: positive, negative, neutral
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train_model(model, dataloader, loss_fn, optimizer, device, n_epochs=3)
