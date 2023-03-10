import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load data from CSV file
df = pd.read_csv('train.csv')

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')


# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_1 = str(self.data.iloc[idx, 1])
        text_2 = str(self.data.iloc[idx, 2])
        label = self.data.iloc[idx, 5]

        encoding = self.tokenizer(text_1, text_2, padding='max_length', truncation=True, max_length=128,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Create dataset and dataloader
dataset = CustomDataset(df, tokenizer)
val_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model
NUM_LABELS = 3  # number of labels for classification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=NUM_LABELS)

# Train loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    average_loss = total_loss / len(dataset)
    print(f'Epoch {epoch}: Train loss: {average_loss:.4f}')

    # Evaluate
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += input_ids.size(0)
    accuracy = total_correct / total_samples
    average_loss = total_loss / total_samples
    print(f'Validation loss: {average_loss:.4f}, Validation accuracy: {accuracy:.4f}')

# Save the model
PATH = 'model.pt'
torch.save(model.state_dict(), PATH)