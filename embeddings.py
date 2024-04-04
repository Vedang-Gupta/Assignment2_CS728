import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import re
import pickle

def split_into_sentences(paragraph):
    sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
    return sentences

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.extend(split_into_sentences(json_obj['text']))
    return data

model = SentenceTransformer('pinecone/bert-retriever-squad2', device = 'cuda')


if __name__ == "__main__":
    i = 1
    base_path = "/home/raavi/SAM-MSCG/wikicheck/wiki-pages/"
    batch_size = 100
    files = os.listdir(base_path)
    for file in tqdm(files[7:]):
        embeddings = []
        sentences = read_jsonl(base_path + file)
        for j in tqdm(range(0, len(sentences), 10)):
            embed = model.encode(sentences[j: min(j + 10, len(sentences))])
            embeddings.extend(embed)
        path = '/home/raavi/SAM-MSCG/wikicheck/wiki-embeddings/' + file[:-6] + '.pkl'
        with open(path, 'wb') as file:
            pickle.dump(embeddings, file)
        # print(embeddings.shape)
        # i += 1





