import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
# Define the 3 labels
LABELS = ["0", "1", "2"]
#0진실 1중립 2 모순

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_1 = str(self.data.iloc[idx, 1])
        text_2 = str(self.data.iloc[idx, 2])

        encoding = self.tokenizer(text_1, text_2, padding='max_length', truncation=True, max_length=128,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
        }

# Load the test data
test_data = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")

# Define the tokenizer and max sequence length
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
MAX_LEN = 128

# Define the custom dataset and dataloader for the test data
test_dataset = CustomDataset(test_data, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the custom model
NUM_LABELS = len(LABELS)
model = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=NUM_LABELS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



file_list = os.listdir("/kaggle/working/")

for i in file_list:
    # Load the saved model weights
    model.load_state_dict(torch.load(f"/kaggle/working/{i}"))

    # Put the model in evaluation mode
    model.eval()

    # Generate predictions for the test data
    preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            preds.extend(predicted.cpu().numpy())

    # Save the predictions in CSV format
    test_data['prediction'] = preds
    test_data=test_data[['id','prediction']]
    test_data.to_csv(f'submission_{i}.csv', index=False)