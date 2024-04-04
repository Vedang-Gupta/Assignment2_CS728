import numpy as np
import torch
from tqdm import tqdm
import json
import os
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader
import pickle

with open("fever_data_train_2.pkl", "rb") as f:
    fever_data_train = pickle.load(f)

with open("fever_data_test_2.pkl", "rb") as f:
    fever_data_test = pickle.load(f)


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

class NLI_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim_evid, label = self.data[idx]
        return claim_evid, label

class NLIModel(torch.nn.Module):
    def __init__(self, model):
        super(NLIModel, self).__init__()
        self.model = model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )
        
    def forward(self, claim_evid):
        input_ids = claim_evid['input_ids']
        attention_mask = claim_evid['attention_mask']
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        logits = self.classifier(pooled_output)
        return logits

train_dataset = NLI_Dataset(fever_data_train)
test_dataset = NLI_Dataset(fever_data_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = NLIModel(bert_model)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(2):
    model.train()
    for claim_evid, label in tqdm(train_loader):
        claim_evid = tokenizer(claim_evid, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        claim_evid = {k: v.to(device) for k, v in claim_evid.items()}
        label = label.to(device)
        optimizer.zero_grad()
        logits = model(claim_evid)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, loss {loss.item()}")
        
    # micro-F1 score
    model.eval()
    all_preds = []
    all_labels = []
    for claim_evid, label in test_loader:
        claim_evid = tokenizer(claim_evid, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        claim_evid = {k: v.to(device) for k, v in claim_evid.items()}
        label = label.to(device)
        logits = model(claim_evid)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    TP = 0
    FP = 0
    FN = 0
    for i in range(3):
        TP += np.sum((all_preds == i) & (all_labels == i))
        FP += np.sum((all_preds == i) & (all_labels != i))
        FN += np.sum((all_preds != i) & (all_labels == i))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Epoch {epoch}, micro f1 {f1}")
    