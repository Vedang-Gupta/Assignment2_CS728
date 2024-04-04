from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import pickle


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append((json_obj['claim']))
    return data


model = SentenceTransformer("pinecone/bert-retriever-squad2", device = 'cuda:1')

if __name__ == "__main__":
    i = 1
    batch_size = 20
    sentences = read_jsonl('/home/raavi/SAM-MSCG/wikicheck/train.jsonl')
    embeddings = []
    for j in tqdm(range(0, len(sentences), 10)):
        embed = model.encode(sentences[j: min(j + 10, len(sentences))])
        embeddings.extend(embed)
    path = '/home/raavi/SAM-MSCG/wikicheck/claims-same-bert.pkl'
    with open(path, 'wb') as file:
        pickle.dump(embeddings, file)
