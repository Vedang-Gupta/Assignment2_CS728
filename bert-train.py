import torch
from transformers import BertTokenizer, BertForMaskedLM
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import random
from tqdm import tqdm


WIKI_DUMP_DIR = "wiki-pages"


def preprocess_wiki_dump(wiki_dump_dir):
    wiki_text = []
    for file_name in os.listdir(wiki_dump_dir):
        with open(os.path.join(wiki_dump_dir, file_name), "r", encoding="utf-8") as f:
            wiki_text.append(json.loads(f.strip()["text"]).read())
    return wiki_text


class WikiDataset(Dataset):
    def __init__(self, wiki_text, device):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.input_ids = []
        self.device = device
        for text in tqdm(wiki_text):
            tokenized_text = self.tokenizer.tokenize(text)
            self.input_ids.append(torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_text)).to(device))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


class BERTForRetrieval(torch.nn.Module):
    def __init__(self):
        super(BERTForRetrieval, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        return self.softmax(outputs.logits)

def train(device):
    wiki_text = preprocess_wiki_dump(WIKI_DUMP_DIR)
    dataset = WikiDataset(wiki_text, device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BERTForRetrieval().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            output = model(batch)
            loss = torch.nn.CrossEntropyLoss()(output, torch.zeros(output.shape[0], dtype=torch.long).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss}")
    torch.save(model.state_dict(), "bert_retriever.pt")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train(device)
