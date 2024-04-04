from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm import tqdm
import json
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pickle

sentence_model = 'all-MiniLM-L6-v2'
dims = 384
device = "cuda"

with open("fever_data_train_1.pkl", "rb") as f:
    fever_data_train = pickle.load(f)

with open("fever_data_test_1.pkl", "rb") as f:
    fever_data_test = pickle.load(f)

class NLI_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim, evidence, label = self.data[idx]
        return claim, evidence, label

    
# shared params for claim and evidence
class NLIModel(torch.nn.Module):
    def __init__(self, model_name):
        super(NLIModel, self).__init__()
        self.encoder = SentenceTransformer(model_name, device=device)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dims * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )

    def forward(self, claim, evidence):
        claim_embedding = self.encoder.encode(claim, convert_to_tensor=True, device=device)
        evidence_embedding = self.encoder.encode(evidence, convert_to_tensor=True, device=device)
        combined_embedding = torch.cat((claim_embedding, evidence_embedding), dim=1)
        logits = self.classifier(combined_embedding)
        return logits
    
# unshared params for claim and evidence

class ClaimEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super(ClaimEncoder, self).__init__()
        self.encoder = SentenceTransformer(model_name)

    def forward(self, claim):
        claim_embedding = self.encoder.encode(claim, convert_to_tensor=True, device=device)
        return claim_embedding
    
class EvidenceEncoder(torch.nn.Module): 
    def __init__(self, model_name):
        super(EvidenceEncoder, self).__init__()
        self.encoder = SentenceTransformer(model_name)

    def forward(self, evidence):
        evidence_embedding = self.encoder.encode(evidence, convert_to_tensor=True, device=device)
        return evidence_embedding
    
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(dims * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )

    def forward(self, claim_embedding, evidence_embedding):
        combined_embedding = torch.cat((claim_embedding, evidence_embedding), dim=1)
        logits = self.classifier(combined_embedding)
        return logits
    
def train_shared(model, train_loader, test_loader, epochs, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    for epoch in range(epochs):
        model.train()
        for claim, evidence, label in tqdm(train_loader):
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(claim, evidence)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} training loss: {loss.item()}")
        test(model, test_loader)

def train_unshared(claim_encoder, evidence_encoder, classifier, train_loader, test_loader, epochs, lr=1e-4):
    optimizer = torch.optim.Adam(list(claim_encoder.parameters()) + list(evidence_encoder.parameters()) + list(classifier.parameters()), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        claim_encoder.train()
        evidence_encoder.train()
        classifier.train()
        for claim, evidence, label in tqdm(train_loader):
            label = label.to(device)
            optimizer.zero_grad()
            claim_embedding = claim_encoder(claim)
            evidence_embedding = evidence_encoder(evidence)
            logits = classifier(claim_embedding, evidence_embedding)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} training loss: {loss.item()}")
        test_unshared(claim_encoder, evidence_encoder, classifier, test_loader)

# test with micro-F1 score 
def test(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    for claim, evidence, label in test_loader:
        label = label.to(device)
        logits = model(claim, evidence)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(label.tolist())
        y_pred.extend(preds.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = 0
    fp = 0
    fn = 0
    for i in range(3):
        tp += np.sum((y_true == i) & (y_pred == i))
        fp += np.sum((y_true != i) & (y_pred == i))
        fn += np.sum((y_true == i) & (y_pred != i))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Test Micro F1: {f1}")

def test_unshared(claim_encoder, evidence_encoder, classifier, test_loader):
    claim_encoder.eval()
    evidence_encoder.eval()
    classifier.eval()
    y_true = []
    y_pred = []
    for claim, evidence, label in test_loader:
        label = label.to(device)
        claim_embedding = claim_encoder(claim)
        evidence_embedding = evidence_encoder(evidence)
        logits = classifier(claim_embedding, evidence_embedding)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(label.tolist())
        y_pred.extend(preds.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = 0
    fp = 0
    fn = 0
    for i in range(3):
        tp += np.sum((y_true == i) & (y_pred == i))
        fp += np.sum((y_true != i) & (y_pred == i))
        fn += np.sum((y_true == i) & (y_pred != i))

    print(tp, fp, fn)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Test Micro F1: {f1}")

if __name__ == "__main__":
    train_dataset = NLI_Dataset(fever_data_train)
    test_dataset = NLI_Dataset(fever_data_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # # shared model
    # shared_model = NLIModel(sentence_model)
    # shared_model.to(device)
    # train_shared(shared_model, train_loader, test_loader, epochs=5)

    # # save model
    # torch.save(shared_model.state_dict(), "shared_model.pth")
    
    # unshared model
    claim_encoder = ClaimEncoder(sentence_model)
    evidence_encoder = EvidenceEncoder(sentence_model)
    classifier = Classifier()
    claim_encoder.to(device)
    evidence_encoder.to(device)
    classifier.to(device)
    train_unshared(claim_encoder, evidence_encoder, classifier, train_loader, test_loader, epochs=3)
    torch.save(claim_encoder.state_dict(), "claim_encoder.pth")
    torch.save(evidence_encoder.state_dict(), "evidence_encoder.pth")
    torch.save(classifier.state_dict(), "classifier.pth")